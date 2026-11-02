<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { onBeforeRouteLeave, onBeforeRouteUpdate, useRoute, useRouter } from 'vue-router'
import { api, formatIDR, type ProductDetail, type RecsResponse } from '../api'
import ProductGrid from '../components/ProductGrid.vue'
import ProductImage from '../components/ProductImage.vue'
import { track } from '../composables/useTracker'
import { useAuth } from '../stores/auth'
import { useCart } from '../stores/cart'

const route = useRoute()
const router = useRouter()
const auth = useAuth()
const cart = useCart()

const item = ref<ProductDetail | null>(null)
const similar = ref<RecsResponse | null>(null)
const qty = ref(1)
const error = ref('')
const added = ref(false)

// book:start dwell
let start = performance.now()
let viewItemId: number | null = null
let viewSurface = 'direct'
let viewPos = 0
let clickSent = true // guard: exactly one click per product view

function beginView() {
  start = performance.now()
  viewItemId = Number(route.params.id)
  viewSurface = String(route.query.src ?? 'direct')
  viewPos = Number(route.query.pos ?? 0)
  clickSent = false
}
function endView() {
  if (clickSent || viewItemId === null) return
  clickSent = true
  track({
    item_id: viewItemId,
    event_type: 'click',
    position: viewPos,
    surface: viewSurface,
    dwell_sec: Math.round(((performance.now() - start) / 1000) * 10) / 10,
  })
}
onBeforeRouteLeave(endView)
onBeforeRouteUpdate((to, from) => {
  if (to.params.id !== from.params.id) endView()
})
window.addEventListener('pagehide', endView)
onUnmounted(() => {
  window.removeEventListener('pagehide', endView)
  endView()
})
// book:end dwell

async function load() {
  const id = route.params.id
  if (!id) return
  beginView()
  error.value = ''
  added.value = false
  qty.value = 1
  item.value = null
  similar.value = null
  try {
    item.value = await api<ProductDetail>(`/products/${id}`)
  } catch (e) {
    error.value = (e as Error).message
    clickSent = true
    return
  }
  try {
    similar.value = await api<RecsResponse>(`/products/${id}/similar?k=12`)
  } catch {
    /* similar items are optional */
  }
}
watch(() => route.params.id, load)
onMounted(load)

async function addToCart() {
  if (!auth.user) {
    void router.push({ path: '/login', query: { next: route.fullPath } })
    return
  }
  error.value = ''
  try {
    await cart.add(Number(route.params.id), qty.value)
    added.value = true
  } catch (e) {
    error.value = (e as Error).message
  }
}
</script>

<template>
  <p v-if="error && !item" class="rounded bg-red-100 p-2 text-sm text-red-700">{{ error }}</p>
  <template v-if="item">
    <div class="grid gap-6 rounded-lg bg-white p-4 shadow-sm md:grid-cols-[320px_1fr]">
      <ProductImage :id="item.id" :brand="item.brand" :category-id="item.category_id" />
      <div>
        <h1 class="text-2xl font-semibold">{{ item.title }}</h1>
        <p class="text-sm text-gray-500">
          {{ item.brand }} · {{ item.category_name }} · ⭐ {{ item.rating_avg.toFixed(1) }} ({{ item.rating_count }})
        </p>
        <div class="mt-3 text-3xl font-bold text-orange-600">{{ formatIDR(item.final_price) }}</div>
        <div v-if="item.discount_pct > 0" class="text-sm text-gray-400">
          <span class="line-through">{{ formatIDR(item.price) }}</span>
          <span class="ml-2 rounded bg-red-100 px-1 text-red-600">-{{ Math.round(item.discount_pct * 100) }}%</span>
        </div>
        <p v-if="item.free_shipping" class="mt-1 text-sm text-green-600">Free shipping</p>
        <p class="mt-3 text-sm text-gray-700">{{ item.description }}</p>
        <p class="mt-3 text-sm">
          Sold by
          <RouterLink :to="`/shops/${item.shop.id}`" class="font-medium text-orange-600 underline">{{
            item.shop.name
          }}</RouterLink>
          ({{ item.shop.city }})
        </p>
        <div class="mt-4 flex items-center gap-3">
          <template v-if="item.stock > 0">
            <input v-model.number="qty" type="number" min="1" :max="item.stock" class="w-20 rounded border px-2 py-1" />
            <button class="rounded bg-orange-500 px-4 py-2 font-medium text-white" @click="addToCart">Add to cart</button>
            <span class="text-sm text-gray-500">{{ item.stock }} in stock</span>
          </template>
          <span v-else class="font-medium text-red-500">Out of stock</span>
        </div>
        <p v-if="error" class="mt-2 text-sm text-red-600">{{ error }}</p>
        <p v-if="added" class="mt-2 text-sm text-green-600">
          Added to cart. <RouterLink to="/cart" class="underline">View cart</RouterLink>
        </p>
      </div>
    </div>

    <h2 class="mb-2 mt-6 text-lg font-semibold">Similar items</h2>
    <template v-if="similar">
      <p v-if="similar.fallback" class="mb-2 text-xs text-yellow-700">Recommender offline — showing popular items in this category</p>
      <ProductGrid :key="item.id" :items="similar.items" :recs="similar.items" surface="similar" />
    </template>
  </template>
</template>
