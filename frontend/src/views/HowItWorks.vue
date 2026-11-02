<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { ApiError, api, post, RECSYS_DOWN_MSG, type ModelInfo, type MyEvents } from '../api'
import { useAuth } from '../stores/auth'

const auth = useAuth()

const steps = [
  { t: '1. Log events', d: 'Your browser reports impressions, clicks, dwell time; the server logs carts and orders.', g: 'What gets logged and why' },
  { t: '2. Candidates', d: 'Four cheap retrievers pick a few hundred items you might like from thousands.', g: 'Inside the recommender, step by step' },
  { t: '3. Features', d: '46 numbers per (you, item) pair describe affinity, popularity, price, freshness…', g: 'Inside the recommender, step by step' },
  { t: '4. Ranker', d: 'A LightGBM LambdaMART model scores every candidate; SHAP explains the score.', g: 'Reading "Why this?"' },
  { t: '5. Re-rank', d: 'Business rules: drop hidden/bought items, cap per category and shop for variety.', g: 'Inside the recommender, step by step' },
]

// live events
const mine = ref<MyEvents | null>(null)
let poll: number | undefined
async function pollEvents() {
  if (!auth.user) return
  try {
    mine.value = await api<MyEvents>('/me/events?limit=30')
  } catch {
    /* keep last */
  }
}

// model info
const info = ref<ModelInfo | null>(null)
const infoError = ref('')
const METHODS = ['global_popular', 'heuristic_blend', 'lightgbm_ranker', 'lightgbm_rerank']
const COLS = ['recall@20', 'order_recall@20', 'ndcg@20']
const fmt = (v: number | undefined) => (v === undefined ? '–' : v.toFixed(3))
const topFeatures = computed(() => info.value?.features.slice(0, 15) ?? [])
const maxGain = computed(() => Math.max(1e-9, ...topFeatures.value.map((f) => f.gain_share)))

async function loadInfo() {
  infoError.value = ''
  try {
    info.value = await api<ModelInfo>('/recsys/info')
  } catch (e) {
    infoError.value = e instanceof ApiError && e.status === 502 ? RECSYS_DOWN_MSG : (e as Error).message
  }
}

// retrain
const retraining = ref(false)
const retrainMsg = ref('')
const retrainMetrics = ref<Record<string, number> | null>(null)
async function retrain() {
  retraining.value = true
  retrainMsg.value = ''
  try {
    const r = await post<{ model_version: string; metrics: Record<string, number>; train_rows: number }>('/admin/retrain')
    retrainMsg.value = `New model ${r.model_version} trained on ${r.train_rows} rows.`
    retrainMetrics.value = r.metrics
    await loadInfo()
  } catch (e) {
    retrainMsg.value = e instanceof ApiError && e.status === 502 ? RECSYS_DOWN_MSG : (e as Error).message
  } finally {
    retraining.value = false
  }
}

onMounted(() => {
  void loadInfo()
  void pollEvents()
  poll = window.setInterval(pollEvents, 3000)
})
onUnmounted(() => clearInterval(poll))
</script>

<template>
  <h1 class="mb-4 text-2xl font-semibold">How the recommendations work</h1>

  <section class="mb-8 grid gap-2 md:grid-cols-5">
    <div v-for="s in steps" :key="s.t" class="rounded-lg bg-white p-3 text-sm shadow-sm">
      <div class="font-semibold text-orange-600">{{ s.t }}</div>
      <p class="mt-1 text-gray-700">{{ s.d }}</p>
      <p class="mt-2 text-xs text-gray-400">docs/GUIDE.md → “{{ s.g }}”</p>
    </div>
  </section>

  <section class="mb-8">
    <h2 class="mb-2 text-lg font-semibold">Your behaviour, live</h2>
    <p v-if="!auth.user" class="rounded bg-white p-4 text-sm text-gray-500">
      <RouterLink to="/login" class="text-orange-600 underline">Log in</RouterLink> to watch your own events stream in.
    </p>
    <div v-else-if="mine" class="grid gap-4 md:grid-cols-[1fr_240px]">
      <table class="w-full rounded bg-white text-xs shadow-sm">
        <thead class="text-left text-gray-500">
          <tr><th class="p-2">Time</th><th>Event</th><th>Item</th><th>Surface</th><th>Pos</th><th>Dwell</th></tr>
        </thead>
        <tbody>
          <tr v-for="e in mine.events" :key="e.id" class="border-t">
            <td class="p-1.5">{{ new Date(e.ts).toLocaleTimeString() }}</td>
            <td class="font-medium">{{ e.event_type }}</td>
            <td class="max-w-[16rem] truncate">{{ e.item_title }}</td>
            <td>{{ e.surface ?? '' }}</td>
            <td>{{ e.position ?? '' }}</td>
            <td>{{ e.dwell_sec != null ? e.dwell_sec.toFixed(1) : '' }}</td>
          </tr>
          <tr v-if="!mine.events.length"><td colspan="6" class="p-3 text-center text-gray-400">No events yet — browse around.</td></tr>
        </tbody>
      </table>
      <div class="h-fit rounded bg-white p-3 text-xs shadow-sm">
        <div class="mb-1 font-semibold">Recent searches</div>
        <ul>
          <li v-for="(s, i) in mine.searches" :key="i">{{ s.query }}</li>
          <li v-if="!mine.searches.length" class="text-gray-400">none</li>
        </ul>
      </div>
    </div>
  </section>

  <section class="mb-8">
    <h2 class="mb-2 text-lg font-semibold">The model</h2>
    <p v-if="infoError" class="rounded bg-yellow-100 p-3 text-sm text-yellow-800">{{ infoError }}</p>
    <template v-if="info">
      <p class="mb-2 text-sm text-gray-600">
        version <b>{{ info.model_version }}</b> · trained {{ info.trained_at }} · {{ info.events_loaded }} events loaded ·
        {{ info.train_rows }} training rows
      </p>
      <table class="mb-1 rounded bg-white text-sm shadow-sm">
        <thead class="text-left text-gray-500">
          <tr><th class="p-2">Method</th><th v-for="c in COLS" :key="c" class="px-3">{{ c }}</th></tr>
        </thead>
        <tbody>
          <tr v-for="m in METHODS" :key="m" class="border-t">
            <td class="p-2 font-medium">{{ m }}</td>
            <td v-for="c in COLS" :key="c" class="px-3">{{ fmt(info.metrics[`${m}/${c}`]) }}</td>
          </tr>
        </tbody>
      </table>
      <p class="mb-4 text-sm text-gray-600">Candidate recall (ceiling for the ranker): {{ fmt(info.metrics['candidate_recall']) }}</p>

      <h3 class="mb-1 font-medium">Top 15 features by gain share</h3>
      <div class="space-y-1.5">
        <div v-for="f in topFeatures" :key="f.name" class="rounded bg-white p-2 text-xs shadow-sm">
          <div class="flex items-center gap-2">
            <span class="w-52 shrink-0 font-mono">{{ f.name }}</span>
            <div class="h-3 flex-1 rounded bg-gray-100">
              <div class="h-3 rounded bg-orange-400" :style="{ width: (f.gain_share / maxGain) * 100 + '%' }" />
            </div>
            <span class="w-14 text-right">{{ (f.gain_share * 100).toFixed(1) }}%</span>
          </div>
          <div class="mt-0.5 text-gray-500"><span class="rounded bg-gray-100 px-1">{{ f.group }}</span> {{ f.description }}</div>
        </div>
      </div>
    </template>
  </section>

  <section v-if="auth.user?.is_admin" class="mb-8">
    <h2 class="mb-2 text-lg font-semibold">Admin</h2>
    <button class="flex items-center gap-2 rounded bg-orange-500 px-4 py-2 text-white disabled:opacity-60" :disabled="retraining" @click="retrain">
      <span v-if="retraining" class="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
      {{ retraining ? 'Training…' : 'Retrain now' }}
    </button>
    <p v-if="retrainMsg" class="mt-2 text-sm">{{ retrainMsg }}</p>
    <table v-if="retrainMetrics" class="mt-2 rounded bg-white text-xs shadow-sm">
      <tr v-for="(v, k) in retrainMetrics" :key="k" class="border-t"><td class="p-1.5 font-mono">{{ k }}</td><td class="px-3">{{ v.toFixed(4) }}</td></tr>
    </table>
  </section>
</template>
