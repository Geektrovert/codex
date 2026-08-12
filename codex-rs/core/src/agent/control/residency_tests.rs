use super::V2Residency;
use crate::StartThreadOptions;
use crate::ThreadManager;
use crate::agent::AgentControl;
use crate::agent::AgentStatus;
use crate::agent::registry::AgentMetadata;
use crate::agent_communication::AgentCommunicationContext;
use crate::agent_communication::AgentCommunicationKind;
use crate::codex_thread::CodexThread;
use crate::config::Config;
use crate::config::test_config;
use crate::thread_manager::ThreadManagerState;
use crate::thread_manager::build_models_manager;
use crate::thread_manager::thread_store_from_config;
use codex_features::Feature;
use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_protocol::AgentPath;
use codex_protocol::ThreadId;
use codex_protocol::error::CodexErrorDetails;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::InterAgentCommunication;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;
use codex_protocol::protocol::ThreadSource;
use codex_protocol::protocol::TurnAbortReason;
use codex_protocol::protocol::TurnAbortedEvent;
use codex_protocol::protocol::TurnCompleteEvent;
use pretty_assertions::assert_eq;
use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::Notify;
use tokio::time::Duration;
use tokio::time::timeout;

struct BlockingThreadStop {
    target_thread_id: Mutex<Option<String>>,
    entered: Notify,
    proceed: Notify,
}

impl codex_extension_api::ThreadLifecycleContributor<Config> for BlockingThreadStop {
    fn on_thread_stop<'a>(
        &'a self,
        input: codex_extension_api::ThreadStopInput<'a>,
    ) -> codex_extension_api::ExtensionFuture<'a, ()> {
        Box::pin(async move {
            let should_block = self
                .target_thread_id
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_deref()
                == Some(input.thread_store.level_id());
            if should_block {
                self.entered.notify_one();
                self.proceed.notified().await;
            }
        })
    }
}

#[test]
fn only_matching_submission_clears_release_protection() {
    let residency = V2Residency::default();
    let thread_id = ThreadId::new();
    residency.begin_submission(thread_id, "pending-submission");

    residency.finish_submission(thread_id, "unrelated-submission");
    assert!(residency.has_pending_submission(thread_id));

    residency.finish_submission(thread_id, "pending-submission");
    assert!(!residency.has_pending_submission(thread_id));
}

#[tokio::test]
async fn residency_slot_reservation_unloads_oldest_idle_v2_agent() {
    let mut config = test_config().await;
    let _ = config.features.enable(Feature::MultiAgentV2);
    config.multi_agent_v2.max_concurrent_threads_per_session = 2;
    let temp_home = tempfile::tempdir().expect("create temp home");
    config.codex_home = temp_home.path().to_path_buf().try_into().unwrap();
    config.cwd = temp_home.path().to_path_buf().try_into().unwrap();
    let manager = ThreadManager::with_models_provider_and_home_for_tests(
        CodexAuth::from_api_key("dummy"),
        config.model_provider.clone(),
        config.codex_home.to_path_buf(),
        Arc::new(codex_exec_server::EnvironmentManager::default_for_tests()),
    );
    let root = manager
        .start_thread(StartThreadOptions::new(config.clone()))
        .await
        .expect("start root thread");
    let control = manager.agent_control();
    let state = control.upgrade().expect("thread manager should be live");

    let first_slot = control
        .reserve_v2_residency_slot(&state, &config, /*protected_thread_id*/ None)
        .await
        .expect("first resident slot");
    let first =
        spawn_v2_subagent(&control, &state, config.clone(), root.thread_id, "worker-1").await;
    first_slot.commit(first.thread_id);
    mark_thread_completed(first.thread.as_ref()).await;

    let second_slot = control
        .reserve_v2_residency_slot(&state, &config, /*protected_thread_id*/ None)
        .await
        .expect("second resident slot should evict the first idle agent");
    match manager.get_thread(first.thread_id).await {
        Err(err) => match err.details() {
            CodexErrorDetails::ThreadNotFound(thread_id) => assert_eq!(*thread_id, first.thread_id),
            _ => panic!("expected evicted thread to be missing, got {err:?}"),
        },
        Ok(_) => panic!("expected evicted thread to be missing"),
    }
    let second = spawn_v2_subagent(&control, &state, config, root.thread_id, "worker-2").await;
    second_slot.commit(second.thread_id);

    assert!(manager.get_thread(root.thread_id).await.is_ok());
    assert!(manager.get_thread(second.thread_id).await.is_ok());
}

#[tokio::test]
async fn released_interrupted_v2_agent_reloads_with_its_last_status() {
    let mut config = test_config().await;
    let _ = config.features.enable(Feature::MultiAgentV2);
    let temp_home = tempfile::tempdir().expect("create temp home");
    config.codex_home = temp_home.path().to_path_buf().try_into().unwrap();
    config.cwd = temp_home.path().to_path_buf().try_into().unwrap();
    let manager = ThreadManager::with_models_provider_and_home_for_tests(
        CodexAuth::from_api_key("dummy"),
        config.model_provider.clone(),
        config.codex_home.to_path_buf(),
        Arc::new(codex_exec_server::EnvironmentManager::default_for_tests()),
    );
    let root = manager
        .start_thread(StartThreadOptions::new(config.clone()))
        .await
        .expect("start root thread");
    let control = manager.agent_control();
    control.register_session_root(root.thread_id, /*current_parent_thread_id*/ None);
    let state = control.upgrade().expect("thread manager should be live");
    let slot = control
        .reserve_v2_residency_slot(&state, &config, /*protected_thread_id*/ None)
        .await
        .expect("resident slot");
    let worker_path = AgentPath::try_from("/root/worker").expect("valid worker path");
    let worker = spawn_registered_v2_subagent(
        &control,
        &state,
        config.clone(),
        root.thread_id,
        worker_path,
    )
    .await;
    slot.commit(worker.thread_id);
    mark_thread_interrupted(worker.thread.as_ref()).await;

    control
        .release_v2_agent(worker.thread_id)
        .await
        .expect("release terminal worker");
    assert!(manager.get_thread(worker.thread_id).await.is_err());

    control
        .ensure_v2_agent_loaded(config, worker.thread_id)
        .await
        .expect("reload released worker");
    let reloaded = manager
        .get_thread(worker.thread_id)
        .await
        .expect("worker should reload with the same identity");
    assert_eq!(reloaded.agent_status().await, AgentStatus::Interrupted);
}

#[tokio::test]
async fn followup_waits_for_release_and_reloads_the_agent() {
    let mut config = test_config().await;
    let _ = config.features.enable(Feature::MultiAgentV2);
    let temp_home = tempfile::tempdir().expect("create temp home");
    config.codex_home = temp_home.path().to_path_buf().try_into().unwrap();
    config.cwd = temp_home.path().to_path_buf().try_into().unwrap();
    let blocker = Arc::new(BlockingThreadStop {
        target_thread_id: Mutex::new(None),
        entered: Notify::new(),
        proceed: Notify::new(),
    });
    let mut extensions = codex_extension_api::ExtensionRegistryBuilder::new();
    extensions.thread_lifecycle_contributor(blocker.clone());
    let auth_manager = AuthManager::from_auth_for_testing(CodexAuth::from_api_key("dummy"));
    let manager = ThreadManager::new(
        &config,
        Arc::clone(&auth_manager),
        build_models_manager(&config, auth_manager),
        crate::CodexAppsToolsCache::default(),
        SessionSource::Exec,
        Arc::new(codex_exec_server::EnvironmentManager::default_for_tests()),
        Arc::new(extensions.build()),
        Arc::new(crate::test_support::EmptyUserInstructionsProvider),
        /*analytics_events_client*/ None,
        thread_store_from_config(&config, /*state_db*/ None),
        /*agent_graph_store*/ None,
        "residency-race-test".to_string(),
        /*attestation_provider*/ None,
        /*external_time_provider*/ None,
    );
    let root = manager
        .start_thread(StartThreadOptions::new(config.clone()))
        .await
        .expect("start root thread");
    let control = manager.agent_control();
    control.register_session_root(root.thread_id, /*current_parent_thread_id*/ None);
    let state = control.upgrade().expect("thread manager should be live");
    let slot = control
        .reserve_v2_residency_slot(&state, &config, /*protected_thread_id*/ None)
        .await
        .expect("resident slot");
    let worker_path = AgentPath::try_from("/root/worker").expect("valid worker path");
    let worker = spawn_registered_v2_subagent(
        &control,
        &state,
        config.clone(),
        root.thread_id,
        worker_path.clone(),
    )
    .await;
    slot.commit(worker.thread_id);
    mark_thread_completed(worker.thread.as_ref()).await;
    *blocker
        .target_thread_id
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(worker.thread_id.to_string());

    let releasing = tokio::spawn({
        let control = control.clone();
        async move { control.release_v2_agent(worker.thread_id).await }
    });
    timeout(Duration::from_secs(5), blocker.entered.notified())
        .await
        .expect("release should enter thread shutdown");

    let followup_started = Arc::new(Notify::new());
    let followup = tokio::spawn({
        let control = control.clone();
        let followup_started = Arc::clone(&followup_started);
        async move {
            followup_started.notify_one();
            control
                .send_inter_agent_communication_to_v2(
                    config,
                    worker.thread_id,
                    InterAgentCommunication::new(
                        AgentPath::root(),
                        worker_path,
                        Vec::new(),
                        "follow up after release".to_string(),
                        /*trigger_turn*/ true,
                    ),
                    AgentCommunicationContext::new(
                        AgentCommunicationKind::Followup,
                        root.thread_id,
                    ),
                    /*parent_turn_id*/ None,
                    /*root_turn_id*/ None,
                )
                .await
        }
    });
    followup_started.notified().await;
    assert!(
        !followup.is_finished(),
        "follow-up must wait for the in-progress release"
    );
    blocker.proceed.notify_one();
    releasing
        .await
        .expect("release task should finish")
        .expect("release should succeed");
    followup
        .await
        .expect("follow-up task should finish")
        .expect("follow-up should be admitted");

    assert!(
        manager.get_thread(worker.thread_id).await.is_ok(),
        "a follow-up admitted during release must run on a reloaded worker"
    );
}

async fn spawn_v2_subagent(
    control: &AgentControl,
    state: &Arc<ThreadManagerState>,
    config: Config,
    parent_thread_id: ThreadId,
    label: &str,
) -> crate::thread_manager::NewThread {
    state
        .spawn_new_thread_with_source(
            config,
            control.clone(),
            SessionSource::SubAgent(SubAgentSource::Other(label.to_string())),
            /*history_mode*/ None,
            Some(parent_thread_id),
            /*forked_from_thread_id*/ None,
            Some(ThreadSource::Subagent),
            /*metrics_service_name*/ None,
            /*inherited_environments*/ None,
            /*inherited_exec_policy*/ None,
            /*environments*/ None,
        )
        .await
        .expect("spawn v2 subagent")
}

async fn spawn_registered_v2_subagent(
    control: &AgentControl,
    state: &Arc<ThreadManagerState>,
    config: Config,
    parent_thread_id: ThreadId,
    agent_path: AgentPath,
) -> crate::thread_manager::NewThread {
    let thread = state
        .spawn_new_thread_with_source(
            config,
            control.clone(),
            SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
                parent_thread_id,
                depth: 1,
                agent_path: Some(agent_path.clone()),
                agent_nickname: None,
                agent_role: None,
            }),
            /*history_mode*/ None,
            Some(parent_thread_id),
            /*forked_from_thread_id*/ None,
            Some(ThreadSource::Subagent),
            /*metrics_service_name*/ None,
            /*inherited_environments*/ None,
            /*inherited_exec_policy*/ None,
            /*environments*/ None,
        )
        .await
        .expect("spawn registered v2 subagent");
    let mut reservation = control
        .state
        .reserve_spawn_slot(/*max_threads*/ None)
        .expect("reserve registry slot");
    reservation
        .reserve_agent_path(&agent_path)
        .expect("reserve worker path");
    reservation.commit(AgentMetadata {
        agent_id: Some(thread.thread_id),
        agent_path: Some(agent_path),
        ..Default::default()
    });
    thread
}

async fn mark_thread_completed(thread: &CodexThread) {
    let turn = thread.session.new_default_turn().await;
    thread
        .session
        .send_event(
            turn.as_ref(),
            EventMsg::TurnComplete(TurnCompleteEvent {
                turn_id: turn.sub_id.clone(),
                started_at: None,
                last_agent_message: Some("done".to_string()),
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            }),
        )
        .await;
    clear_active_turn(thread).await;
}

async fn mark_thread_interrupted(thread: &CodexThread) {
    let turn = thread.session.new_default_turn().await;
    thread
        .session
        .send_event(
            turn.as_ref(),
            EventMsg::TurnAborted(TurnAbortedEvent {
                turn_id: Some(turn.sub_id.clone()),
                started_at: None,
                reason: TurnAbortReason::Interrupted,
                completed_at: None,
                duration_ms: None,
            }),
        )
        .await;
    clear_active_turn(thread).await;
}

async fn clear_active_turn(thread: &CodexThread) {
    // The fixture has no task runner to clear the turn after the terminal event.
    *thread.session.active_turn.lock().await = None;
}
