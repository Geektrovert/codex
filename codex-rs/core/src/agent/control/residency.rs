use super::AgentControl;
use crate::agent::AgentStatus;
use crate::codex_thread::CodexThread;
use crate::config::Config;
use crate::thread_manager::ThreadManagerState;
use codex_protocol::ThreadId;
use codex_protocol::error::CodexErr;
use codex_protocol::error::CodexErrorDetails;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::protocol::MultiAgentVersion;
use codex_protocol::protocol::SessionSource;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::OwnedMutexGuard;
use tracing::warn;

#[derive(Default)]
pub(super) struct V2Residency {
    state: Mutex<V2ResidencyState>,
}

#[derive(Default)]
struct V2ResidencyState {
    residents: VecDeque<ThreadId>,
    pending_slots: usize,
    pending_submissions: HashMap<ThreadId, HashSet<String>>,
    lifecycle_gates: HashMap<ThreadId, Weak<AsyncMutex<()>>>,
}

pub(super) struct V2ResidencySlot {
    residency: Arc<V2Residency>,
    active: bool,
}

impl V2ResidencySlot {
    pub(super) fn commit(mut self, thread_id: ThreadId) {
        self.residency.commit_slot(thread_id);
        self.active = false;
    }
}

impl Drop for V2ResidencySlot {
    fn drop(&mut self) {
        if self.active {
            self.residency.release_pending_slot();
        }
    }
}

impl AgentControl {
    pub(super) async fn reserve_v2_residency_slot(
        &self,
        state: &Arc<ThreadManagerState>,
        config: &Config,
        protected_thread_id: Option<ThreadId>,
    ) -> CodexResult<V2ResidencySlot> {
        let capacity = config
            .effective_agent_max_threads(MultiAgentVersion::V2)
            .unwrap_or(usize::MAX);
        Arc::clone(&self.v2_residency)
            .reserve_slot(state, capacity, protected_thread_id)
            .await
    }

    pub(super) async fn touch_loaded_v2_residency(
        &self,
        state: &Arc<ThreadManagerState>,
        thread_id: ThreadId,
    ) {
        if let Ok(thread) = state.get_thread(thread_id).await
            && is_resident_candidate(thread.as_ref())
        {
            self.v2_residency.touch(thread_id);
        }
    }

    pub(super) fn forget_v2_residency(&self, thread_id: ThreadId) {
        self.v2_residency.remove(thread_id);
    }

    pub(super) fn begin_v2_submission(&self, thread_id: ThreadId, submission_id: &str) {
        self.v2_residency.begin_submission(thread_id, submission_id);
    }

    pub(crate) fn finish_v2_submission(&self, thread_id: ThreadId, submission_id: &str) {
        self.v2_residency
            .finish_submission(thread_id, submission_id);
    }

    /// Unload a terminal v2 agent's in-process session while preserving its durable identity.
    ///
    /// A later message or follow-up task can reload the agent from its rollout. This is distinct
    /// from `close_agent`, which closes the persisted spawn edge and makes the agent unavailable.
    pub(crate) async fn release_v2_agent(&self, thread_id: ThreadId) -> CodexResult<AgentStatus> {
        let _lifecycle_guard = self.lock_v2_lifecycle(thread_id).await;
        let state = self.upgrade()?;
        let thread = match state.get_thread(thread_id).await {
            Ok(thread) => thread,
            Err(err)
                if self.state.agent_metadata_for_thread(thread_id).is_some()
                    && matches!(err.details(), CodexErrorDetails::ThreadNotFound(_)) =>
            {
                self.forget_v2_residency(thread_id);
                return Ok(AgentStatus::NotFound);
            }
            Err(err) => return Err(err),
        };
        if !is_resident_candidate(thread.as_ref()) {
            return Err(CodexErr::UnsupportedOperation(
                "only spawned MultiAgentV2 agents can be released".to_string(),
            ));
        }
        let previous_status = thread.agent_status().await;
        if self.v2_residency.has_pending_submission(thread_id)
            || !is_releasable_with_status(thread.as_ref(), &previous_status).await
        {
            return Err(CodexErr::UnsupportedOperation(
                "agent must be completed or interrupted with no pending messages before it can be released"
                    .to_string(),
            ));
        }

        unload_v2_resident(&state, thread_id, thread.as_ref()).await?;
        self.forget_v2_residency(thread_id);
        Ok(previous_status)
    }

    pub(super) async fn lock_v2_lifecycle(&self, thread_id: ThreadId) -> OwnedMutexGuard<()> {
        self.v2_residency.lock_lifecycle(thread_id).await
    }
}

impl V2Residency {
    async fn reserve_slot(
        self: Arc<Self>,
        manager: &Arc<ThreadManagerState>,
        capacity: usize,
        protected_thread_id: Option<ThreadId>,
    ) -> CodexResult<V2ResidencySlot> {
        loop {
            if self.try_reserve_pending_slot(capacity) {
                return Ok(V2ResidencySlot {
                    residency: self,
                    active: true,
                });
            }
            if !self
                .try_unload_one_resident(manager, protected_thread_id)
                .await
            {
                return Err(CodexErr::new(CodexErrorDetails::AgentLimitReached {
                    max_threads: capacity,
                }));
            }
        }
    }

    fn try_reserve_pending_slot(&self, capacity: usize) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.residents.len().saturating_add(state.pending_slots) >= capacity {
            return false;
        }
        state.pending_slots += 1;
        true
    }

    async fn try_unload_one_resident(
        &self,
        manager: &Arc<ThreadManagerState>,
        protected_thread_id: Option<ThreadId>,
    ) -> bool {
        let candidates_to_scan = self.resident_count();
        for _ in 0..candidates_to_scan {
            let Some(candidate_thread_id) = self.pop_lru_candidate(protected_thread_id) else {
                return false;
            };
            let _lifecycle_guard = self.lock_lifecycle(candidate_thread_id).await;
            let Some(candidate_thread) = manager
                .get_thread(candidate_thread_id)
                .await
                .ok()
                .filter(|thread| is_resident_candidate(thread))
            else {
                self.remove(candidate_thread_id);
                continue;
            };
            if self.has_pending_submission(candidate_thread_id)
                || !is_unloadable(candidate_thread.as_ref()).await
            {
                self.touch(candidate_thread_id);
                continue;
            }
            if let Err(err) =
                unload_v2_resident(manager, candidate_thread_id, candidate_thread.as_ref()).await
            {
                warn!(
                    "failed to shut down v2 resident thread before unloading {candidate_thread_id}: {err}"
                );
                self.touch(candidate_thread_id);
                continue;
            }
            self.remove(candidate_thread_id);
            return true;
        }
        false
    }

    async fn lock_lifecycle(&self, thread_id: ThreadId) -> OwnedMutexGuard<()> {
        let gate = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state
                .lifecycle_gates
                .retain(|_, gate| gate.strong_count() > 0);
            state
                .lifecycle_gates
                .get(&thread_id)
                .and_then(Weak::upgrade)
                .unwrap_or_else(|| {
                    let gate = Arc::new(AsyncMutex::new(()));
                    state
                        .lifecycle_gates
                        .insert(thread_id, Arc::downgrade(&gate));
                    gate
                })
        };
        gate.lock_owned().await
    }

    fn resident_count(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .residents
            .len()
    }

    fn pop_lru_candidate(&self, protected_thread_id: Option<ThreadId>) -> Option<ThreadId> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let candidates_to_scan = state.residents.len();
        for _ in 0..candidates_to_scan {
            let candidate_thread_id = state.residents.pop_front()?;
            if Some(candidate_thread_id) == protected_thread_id {
                state.residents.push_back(candidate_thread_id);
                continue;
            }
            return Some(candidate_thread_id);
        }
        None
    }

    fn touch(&self, thread_id: ThreadId) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        touch_resident(&mut state.residents, thread_id);
    }

    fn remove(&self, thread_id: ThreadId) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state
            .residents
            .retain(|resident_thread_id| *resident_thread_id != thread_id);
        state.pending_submissions.remove(&thread_id);
    }

    fn begin_submission(&self, thread_id: ThreadId, submission_id: &str) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending_submissions
            .entry(thread_id)
            .or_default()
            .insert(submission_id.to_string());
    }

    fn finish_submission(&self, thread_id: ThreadId, submission_id: &str) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(pending_ids) = state.pending_submissions.get_mut(&thread_id) else {
            return;
        };
        pending_ids.remove(submission_id);
        if pending_ids.is_empty() {
            state.pending_submissions.remove(&thread_id);
        }
    }

    fn has_pending_submission(&self, thread_id: ThreadId) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending_submissions
            .contains_key(&thread_id)
    }

    fn commit_slot(&self, thread_id: ThreadId) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.pending_slots = state.pending_slots.saturating_sub(1);
        touch_resident(&mut state.residents, thread_id);
    }

    fn release_pending_slot(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.pending_slots = state.pending_slots.saturating_sub(1);
    }
}

fn touch_resident(residents: &mut VecDeque<ThreadId>, thread_id: ThreadId) {
    residents.retain(|resident_thread_id| *resident_thread_id != thread_id);
    residents.push_back(thread_id);
}

fn is_resident_candidate(thread: &CodexThread) -> bool {
    thread.multi_agent_version() == Some(MultiAgentVersion::V2)
        && is_v2_resident_session_source(&thread.session_source)
}

pub(super) fn is_v2_resident_session_source(session_source: &SessionSource) -> bool {
    matches!(session_source, SessionSource::SubAgent(_))
}

async fn is_unloadable(thread: &CodexThread) -> bool {
    let status = thread.agent_status().await;
    is_unloadable_with_status(thread, &status).await
}

async fn is_unloadable_with_status(thread: &CodexThread, status: &AgentStatus) -> bool {
    matches!(
        status,
        AgentStatus::Completed(_) | AgentStatus::Errored(_) | AgentStatus::Interrupted
    ) && is_idle(thread).await
}

async fn is_releasable_with_status(thread: &CodexThread, status: &AgentStatus) -> bool {
    matches!(status, AgentStatus::Completed(_) | AgentStatus::Interrupted) && is_idle(thread).await
}

async fn is_idle(thread: &CodexThread) -> bool {
    thread.session.active_turn.lock().await.is_none()
        && !thread.session.input_queue.has_pending_mailbox_items().await
}

async fn unload_v2_resident(
    manager: &Arc<ThreadManagerState>,
    thread_id: ThreadId,
    thread: &CodexThread,
) -> CodexResult<()> {
    thread.ensure_rollout_materialized().await;
    thread.shutdown_and_wait().await?;
    let _ = manager.remove_thread(&thread_id).await;
    Ok(())
}

#[cfg(test)]
#[path = "residency_tests.rs"]
mod tests;
