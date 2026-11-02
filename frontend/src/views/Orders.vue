<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api, formatIDR, type Order } from '../api'

const orders = ref<Order[] | null>(null)
const error = ref('')
const badge: Record<string, string> = {
  paid: 'bg-blue-100 text-blue-700',
  shipped: 'bg-yellow-100 text-yellow-800',
  completed: 'bg-green-100 text-green-700',
}
onMounted(async () => {
  try {
    orders.value = await api<Order[]>('/orders')
  } catch (e) {
    error.value = (e as Error).message
  }
})
</script>

<template>
  <h1 class="mb-3 text-xl font-semibold">My orders</h1>
  <p v-if="error" class="rounded bg-red-100 p-2 text-sm text-red-700">{{ error }}</p>
  <p v-if="orders && !orders.length" class="rounded bg-white p-8 text-center text-gray-500">No orders yet.</p>
  <div class="space-y-3">
    <div v-for="o in orders ?? []" :key="o.id" class="rounded bg-white p-4 shadow-sm">
      <div class="flex items-center justify-between">
        <div class="font-medium">Order #{{ o.id }} · {{ o.shop_name }}</div>
        <span class="rounded px-2 py-0.5 text-xs font-medium" :class="badge[o.status]">{{ o.status }}</span>
      </div>
      <div class="text-xs text-gray-500">{{ new Date(o.created_at).toLocaleString() }}</div>
      <ul class="mt-2 text-sm">
        <li v-for="l in o.items" :key="l.item_id" class="flex justify-between">
          <span>{{ l.quantity }} × {{ l.title }}</span><span>{{ formatIDR(l.unit_price * l.quantity) }}</span>
        </li>
      </ul>
      <div class="mt-2 border-t pt-1 text-right text-sm">
        Shipping {{ formatIDR(o.shipping_fee) }} · <b>Total {{ formatIDR(o.total) }}</b>
      </div>
    </div>
  </div>
</template>
