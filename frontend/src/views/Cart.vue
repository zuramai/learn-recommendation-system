<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { formatIDR } from '../api'
import ProductImage from '../components/ProductImage.vue'
import { useCart } from '../stores/cart'

const cart = useCart()
const error = ref('')

async function run(fn: () => Promise<void>) {
  error.value = ''
  try {
    await fn()
  } catch (e) {
    error.value = (e as Error).message
  }
}
onMounted(() => void run(() => cart.fetch()))
</script>

<template>
  <h1 class="mb-3 text-xl font-semibold tracking-tight">Shopping cart</h1>
  <p v-if="error" class="mb-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{{ error }}</p>
  <div v-if="!cart.cart || !cart.cart.lines.length" class="card p-8 text-center text-neutral-500">
    Your cart is empty. <RouterLink to="/" class="text-primary-600 hover:text-primary-700 hover:underline">Keep shopping</RouterLink>
  </div>
  <template v-else>
    <div class="space-y-2">
      <div v-for="l in cart.cart.lines" :key="l.item.id" class="flex items-center gap-4 card p-3">
        <div class="w-20 shrink-0">
          <ProductImage :id="l.item.id" :brand="l.item.brand" :category-id="l.item.category_id" />
        </div>
        <div class="flex-1">
          <RouterLink :to="`/products/${l.item.id}?src=cart&pos=0`" class="font-medium hover:underline">{{ l.item.title }}</RouterLink>
          <div class="text-sm text-neutral-500">{{ l.item.shop_name }} · {{ formatIDR(l.item.final_price) }}</div>
        </div>
        <div class="flex items-center gap-2">
          <button class="btn-secondary !h-7 !w-7 !p-0" @click="run(() => cart.update(l.item.id, l.quantity - 1))">−</button>
          <span class="w-6 text-center">{{ l.quantity }}</span>
          <button class="btn-secondary !h-7 !w-7 !p-0" @click="run(() => cart.update(l.item.id, l.quantity + 1))">+</button>
        </div>
        <div class="w-28 text-right font-semibold">{{ formatIDR(l.line_total) }}</div>
        <button class="text-sm text-rose-500" @click="run(() => cart.remove(l.item.id))">Remove</button>
      </div>
    </div>
    <div class="mt-4 ml-auto max-w-sm space-y-1 card p-4">
      <div class="flex justify-between"><span>Subtotal</span><span>{{ formatIDR(cart.cart.subtotal) }}</span></div>
      <div class="flex justify-between"><span>Shipping</span><span>{{ formatIDR(cart.cart.shipping_fee) }}</span></div>
      <div class="flex justify-between border-t pt-1 text-lg font-bold"><span>Total</span><span>{{ formatIDR(cart.cart.total) }}</span></div>
      <RouterLink to="/checkout" class="btn-primary mt-3 w-full">Checkout</RouterLink>
    </div>
  </template>
</template>
