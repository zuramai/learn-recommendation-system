<script setup lang="ts">
import { watch } from 'vue'
import NavBar from './components/NavBar.vue'
import { useAuth } from './stores/auth'
import { useCart } from './stores/cart'
import { loadCategories } from './categories'

const auth = useAuth()
const cart = useCart()
void loadCategories()
watch(
  () => auth.user?.id,
  () => void cart.fetch().catch(() => {}),
  { immediate: true },
)
</script>

<template>
  <NavBar />
  <main class="mx-auto max-w-6xl px-4 py-6">
    <RouterView />
  </main>
</template>
