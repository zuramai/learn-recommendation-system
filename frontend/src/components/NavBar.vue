<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuth } from '../stores/auth'
import { useCart } from '../stores/cart'

const auth = useAuth()
const cart = useCart()
const route = useRoute()
const router = useRouter()
const q = ref(typeof route.query.q === 'string' ? route.query.q : '')

function search() {
  void router.push({ path: '/search', query: q.value.trim() ? { q: q.value.trim() } : {} })
}
async function logout() {
  await auth.logout()
  cart.cart = null
  void router.push('/')
}
</script>

<template>
  <header class="sticky top-0 z-10 bg-orange-500 text-white shadow">
    <div class="mx-auto flex max-w-6xl items-center gap-4 px-4 py-3">
      <RouterLink to="/" class="text-xl font-bold">🛍️ Marketplace</RouterLink>
      <form class="flex-1" @submit.prevent="search">
        <input
          v-model="q"
          type="search"
          placeholder="Search products or categories"
          class="w-full rounded px-3 py-1.5 text-gray-900 outline-none"
        />
      </form>
      <nav class="flex items-center gap-4 text-sm">
        <RouterLink to="/how-it-works" class="hover:underline">How it works</RouterLink>
        <RouterLink to="/seller" class="hover:underline">Seller Center</RouterLink>
        <RouterLink to="/cart" class="hover:underline">
          🛒 Cart
          <span v-if="cart.count" class="ml-1 rounded-full bg-white px-1.5 text-xs font-semibold text-orange-600">{{
            cart.count
          }}</span>
        </RouterLink>
        <template v-if="auth.user">
          <RouterLink to="/orders" class="hover:underline">Orders</RouterLink>
          <span class="font-medium">{{ auth.user.name }}</span>
          <button class="rounded bg-white/20 px-2 py-1 hover:bg-white/30" @click="logout">Logout</button>
        </template>
        <template v-else>
          <RouterLink to="/login" class="hover:underline">Login</RouterLink>
          <RouterLink to="/register" class="rounded bg-white px-2 py-1 font-medium text-orange-600">Register</RouterLink>
        </template>
      </nav>
    </div>
  </header>
</template>
