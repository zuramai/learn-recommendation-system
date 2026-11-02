<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api, formatIDR, type Order } from '../api'

const orders = ref<Order[] | null>(null)
const error = ref('')
const badge: Record<string, string> = {
  paid: 'bg-primary-50 text-primary-700',
  shipped: 'bg-neutral-100 text-neutral-700',
  completed: 'bg-emerald-50 text-emerald-700',
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
  <h1 class="mb-3 text-xl font-semibold tracking-tight">My orders</h1>
  <p v-if="error" class="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{{ error }}</p>
  <p v-if="orders && !orders.length" class="card p-8 text-center text-neutral-500">No orders yet.</p>
  <div class="space-y-3">
    <div v-for="o in orders ?? []" :key="o.id" class="card p-4">
      <div class="flex items-center justify-between">
        <div class="font-medium">Order #{{ o.id }} · {{ o.shop_name }}</div>
        <span class="badge" :class="badge[o.status]">{{ o.status }}</span>
      </div>
      <div class="text-xs text-neutral-500">{{ new Date(o.created_at).toLocaleString() }}</div>
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
