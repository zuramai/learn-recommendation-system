<script setup lang="ts">
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { api, type ProductList, type Shop } from '../api'
import ProductGrid from '../components/ProductGrid.vue'

const route = useRoute()
const shop = ref<Shop | null>(null)
const products = ref<ProductList | null>(null)
const error = ref('')

async function load() {
  const id = route.params.id
  if (!id) return
  error.value = ''
  try {
    shop.value = await api<Shop>(`/shops/${id}`)
    products.value = await api<ProductList>(`/products?shop_id=${id}&page_size=60`)
  } catch (e) {
    error.value = (e as Error).message
  }
}
watch(() => route.params.id, load, { immediate: true })
</script>

<template>
  <p v-if="error" class="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{{ error }}</p>
  <template v-if="shop">
    <div class="mb-4 card p-4">
      <h1 class="text-xl font-semibold tracking-tight">
        {{ shop.name }}
        <span v-if="shop.is_mall" class="ml-2 rounded bg-rose-400 px-1.5 py-0.5 text-xs text-white">Mall</span>
      </h1>
      <p class="text-sm text-neutral-500">
        {{ shop.city }} · ⭐ {{ shop.shop_rating.toFixed(1) }} · {{ shop.product_count ?? 0 }} products
      </p>
    </div>
    <ProductGrid v-if="products" :key="String(route.params.id)" :items="products.items" surface="shop" />
  </template>
</template>
