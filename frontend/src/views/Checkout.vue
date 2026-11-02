<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { formatIDR, post } from '../api'
import { useCart } from '../stores/cart'

const cart = useCart()
const router = useRouter()
const address = ref('')
const error = ref('')
const busy = ref(false)

onMounted(() => void cart.fetch().catch((e: Error) => (error.value = e.message)))

async function place() {
  error.value = ''
  busy.value = true
  try {
    await post('/checkout', { shipping_address: address.value })
    await cart.fetch()
    void router.push('/orders')
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <h1 class="mb-3 text-xl font-semibold tracking-tight">Checkout</h1>
  <div v-if="!cart.cart || !cart.cart.lines.length" class="card p-8 text-center text-neutral-500">
    Your cart is empty. <RouterLink to="/" class="text-primary-600 hover:text-primary-700 hover:underline">Keep shopping</RouterLink>
  </div>
  <div v-else class="grid gap-4 md:grid-cols-2">
    <div class="card p-4">
      <label class="mb-1 block text-sm font-medium">Shipping address</label>
      <textarea v-model="address" rows="4" class="input" placeholder="Jl. Contoh No. 1, Jakarta" />
    </div>
    <div class="card p-4">
      <ul class="mb-2 space-y-1 text-sm">
        <li v-for="l in cart.cart.lines" :key="l.item.id" class="flex justify-between">
          <span>{{ l.quantity }} × {{ l.item.title }}</span><span>{{ formatIDR(l.line_total) }}</span>
        </li>
      </ul>
      <div class="flex justify-between text-sm"><span>Subtotal</span><span>{{ formatIDR(cart.cart.subtotal) }}</span></div>
      <div class="flex justify-between text-sm"><span>Shipping</span><span>{{ formatIDR(cart.cart.shipping_fee) }}</span></div>
      <div class="flex justify-between border-t pt-1 text-lg font-bold"><span>Total</span><span>{{ formatIDR(cart.cart.total) }}</span></div>
      <p v-if="error" class="mt-2 text-sm text-rose-600">{{ error }}</p>
      <button class="btn-primary mt-3 w-full" :disabled="busy" @click="place">
        {{ busy ? 'Placing…' : 'Place order' }}
      </button>
      <p class="mt-1 text-center text-xs text-neutral-400">Payment is simulated and always succeeds.</p>
    </div>
  </div>
</template>
