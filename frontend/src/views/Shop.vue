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
  <p v-if="error" class="rounded bg-red-100 p-2 text-sm text-red-700">{{ error }}</p>
  <template v-if="shop">
    <div class="mb-4 rounded-lg bg-white p-4 shadow-sm">
      <h1 class="text-xl font-semibold">
        {{ shop.name }}
        <span v-if="shop.is_mall" class="ml-2 rounded bg-red-500 px-1.5 py-0.5 text-xs text-white">Mall</span>
      </h1>
      <p class="text-sm text-gray-500">
        {{ shop.city }} · ⭐ {{ shop.shop_rating.toFixed(1) }} · {{ shop.product_count ?? 0 }} products
      </p>
    </div>
    <ProductGrid v-if="products" :key="String(route.params.id)" :items="products.items" surface="shop" />
  </template>
</template>
