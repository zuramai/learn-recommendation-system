import { useAuth } from '../stores/auth'

// book:start tracker
export interface TrackEvent {
  item_id: number
  event_type: 'impression' | 'click' | 'hide'
  position?: number
  surface?: string
  dwell_sec?: number
}

const FLUSH_MS = 2000
const FLUSH_AT = 20
let queue: TrackEvent[] = []
let timer: number | null = null

function take(): TrackEvent[] {
  const batch = queue
  queue = []
  if (timer !== null) {
    clearTimeout(timer)
    timer = null
  }
  return batch
}

async function flush() {
  const events = take()
  if (!events.length) return
  try {
    await fetch('/api/events', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ events }),
    })
  } catch {
    /* tracking is best-effort */
  }
}

export function track(e: TrackEvent) {
  if (!useAuth().user) return
  queue.push(e)
  if (queue.length >= FLUSH_AT) void flush()
  else if (timer === null) timer = window.setTimeout(() => void flush(), FLUSH_MS)
}

window.addEventListener('pagehide', () => {
  const events = take()
  if (!events.length) return
  navigator.sendBeacon('/api/events', new Blob([JSON.stringify({ events })], { type: 'application/json' }))
})

export function useTracker() {
  return { track, flush }
}
// book:end tracker
