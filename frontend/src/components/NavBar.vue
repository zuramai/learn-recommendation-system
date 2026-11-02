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
  <header class="sticky top-0 z-10 border-b border-neutral-200 bg-white/90 backdrop-blur">
    <div class="mx-auto flex max-w-6xl items-center gap-6 px-4 py-3">
      <RouterLink to="/" class="flex items-center gap-2 text-lg font-semibold tracking-tight text-neutral-900">
        <span class="flex h-7 w-7 items-center justify-center rounded-lg bg-primary-600 text-sm text-white">M</span>
        Marketplace
      </RouterLink>
      <form class="flex-1" @submit.prevent="search">
        <input
          v-model="q"
          type="search"
          placeholder="Search products or categories"
          class="w-full max-w-xl rounded-full border border-transparent bg-neutral-100 px-4 py-2 text-sm text-neutral-900 placeholder:text-neutral-400 focus:border-primary-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-primary-500/20"
        />
      </form>
      <nav class="flex items-center gap-1 text-sm text-neutral-600">
        <RouterLink to="/how-it-works" class="rounded-lg px-3 py-1.5 hover:bg-neutral-100 hover:text-neutral-900" active-class="!text-primary-700 bg-primary-50">How it works</RouterLink>
        <RouterLink to="/seller" class="rounded-lg px-3 py-1.5 hover:bg-neutral-100 hover:text-neutral-900" active-class="!text-primary-700 bg-primary-50">Seller Center</RouterLink>
        <RouterLink to="/cart" class="flex items-center gap-1.5 rounded-lg px-3 py-1.5 hover:bg-neutral-100 hover:text-neutral-900" active-class="!text-primary-700 bg-primary-50">
          Cart
          <span v-if="cart.count" class="rounded-full bg-primary-600 px-1.5 py-0.5 text-[11px] font-semibold leading-none text-white">{{ cart.count }}</span>
        </RouterLink>
        <template v-if="auth.user">
          <RouterLink to="/orders" class="rounded-lg px-3 py-1.5 hover:bg-neutral-100 hover:text-neutral-900" active-class="!text-primary-700 bg-primary-50">Orders</RouterLink>
          <span class="ml-2 border-l border-neutral-200 pl-4 font-medium text-neutral-900">{{ auth.user.name }}</span>
          <button class="btn-secondary !px-3 !py-1.5" @click="logout">Logout</button>
        </template>
        <template v-else>
          <RouterLink to="/login" class="rounded-lg px-3 py-1.5 hover:bg-neutral-100 hover:text-neutral-900">Login</RouterLink>
          <RouterLink to="/register" class="btn-primary !px-3 !py-1.5">Register</RouterLink>
        </template>
      </nav>
    </div>
  </header>
</template>
