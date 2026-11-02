<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api, type RecsResponse } from '../api'
import { categories } from '../categories'
import ProductGrid from '../components/ProductGrid.vue'
import { useAuth } from '../stores/auth'

const auth = useAuth()
const data = ref<RecsResponse | null>(null)
const loading = ref(false)
const error = ref('')

async function load() {
  loading.value = true
  error.value = ''
  try {
    data.value = await api<RecsResponse>('/recommendations?k=20&surface=home_feed')
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    loading.value = false
  }
}
function hide(id: number) {
  if (data.value) data.value.items = data.value.items.filter((i) => i.id !== id)
}
onMounted(load)
</script>

<template>
  <div class="mb-4 flex flex-wrap gap-2">
    <RouterLink
      v-for="c in categories"
      :key="c.id"
      :to="{ path: '/search', query: { category_id: String(c.id) } }"
      class="rounded-full bg-white px-3 py-1 text-sm shadow-sm hover:bg-orange-50"
      >{{ c.name }}</RouterLink
    >
  </div>

  <div class="mb-2 flex items-center justify-between">
    <h1 class="text-xl font-semibold">{{ auth.user ? 'Recommended for you' : 'Popular now' }}</h1>
    <button class="rounded bg-orange-500 px-3 py-1 text-sm text-white disabled:opacity-50" :disabled="loading" @click="load">
      {{ loading ? 'Loading…' : 'Refresh' }}
    </button>
  </div>

  <p v-if="error" class="mb-3 rounded bg-red-100 p-2 text-sm text-red-700">{{ error }}</p>
  <template v-if="data">
    <div v-if="data.fallback" class="mb-3 rounded bg-yellow-100 p-2 text-sm text-yellow-800">
      Recommender offline — showing popular items
    </div>
    <p v-else class="mb-3 text-xs text-gray-500">
      model {{ data.model_version }} · {{ data.candidates_considered }} candidates · {{ data.latency_ms?.toFixed(0) }} ms
    </p>
    <ProductGrid :items="data.items" :recs="data.items" surface="home_feed" @hide="hide" />
  </template>
</template>
