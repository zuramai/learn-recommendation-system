<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, type ProductList } from '../api'
import ProductGrid from '../components/ProductGrid.vue'

const route = useRoute()
const router = useRouter()
const data = ref<ProductList | null>(null)
const error = ref('')

const q = computed(() => String(route.query.q ?? ''))
const sort = computed(() => String(route.query.sort ?? 'popular'))
const page = computed(() => Number(route.query.page ?? 1))
const pages = computed(() => (data.value ? Math.max(1, Math.ceil(data.value.total / data.value.page_size)) : 1))

async function load() {
  const p = new URLSearchParams()
  if (q.value) p.set('q', q.value)
  if (route.query.category_id) p.set('category_id', String(route.query.category_id))
  p.set('sort', sort.value)
  p.set('page', String(page.value))
  error.value = ''
  try {
    data.value = await api<ProductList>('/products?' + p.toString())
  } catch (e) {
    error.value = (e as Error).message
  }
}
watch(() => route.query, load, { immediate: true })

function setQuery(patch: Record<string, string>) {
  void router.push({ query: { ...route.query, ...patch } })
}
</script>

<template>
  <div class="mb-3 flex items-center justify-between">
    <h1 class="text-xl font-semibold">
      {{ q ? `Results for “${q}”` : 'Products' }}
      <span v-if="data" class="text-sm font-normal text-gray-500">({{ data.total }})</span>
    </h1>
    <select
      :value="sort"
      class="rounded border bg-white px-2 py-1 text-sm"
      @change="setQuery({ sort: ($event.target as HTMLSelectElement).value, page: '1' })"
    >
      <option value="popular">Popular</option>
      <option value="relevance">Relevance</option>
      <option value="newest">Newest</option>
      <option value="price_asc">Price: low to high</option>
      <option value="price_desc">Price: high to low</option>
    </select>
  </div>
  <p v-if="error" class="rounded bg-red-100 p-2 text-sm text-red-700">{{ error }}</p>
  <template v-if="data">
    <p v-if="!data.items.length" class="py-10 text-center text-gray-500">No products found.</p>
    <ProductGrid :key="route.fullPath" :items="data.items" surface="search" />
    <div class="mt-6 flex items-center justify-center gap-3 text-sm">
      <button class="rounded bg-white px-3 py-1 shadow-sm disabled:opacity-40" :disabled="page <= 1" @click="setQuery({ page: String(page - 1) })">
        Prev
      </button>
      <span>Page {{ page }} / {{ pages }}</span>
      <button class="rounded bg-white px-3 py-1 shadow-sm disabled:opacity-40" :disabled="page >= pages" @click="setQuery({ page: String(page + 1) })">
        Next
      </button>
    </div>
  </template>
</template>
