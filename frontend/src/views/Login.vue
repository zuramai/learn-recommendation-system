<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuth } from '../stores/auth'

const auth = useAuth()
const route = useRoute()
const router = useRouter()
const email = ref('')
const password = ref('')
const error = ref('')

async function submit() {
  error.value = ''
  try {
    await auth.login(email.value, password.value)
    void router.push(typeof route.query.next === 'string' ? route.query.next : '/')
  } catch (e) {
    error.value = (e as Error).message
  }
}
</script>

<template>
  <form class="mx-auto max-w-sm space-y-3 rounded bg-white p-6 shadow-sm" @submit.prevent="submit">
    <h1 class="text-xl font-semibold">Login</h1>
    <div class="rounded bg-blue-50 p-2 text-xs text-blue-800">
      Demo: user43@demo.test / password, seller1@demo.test / password, admin@demo.test / password
    </div>
    <input v-model="email" type="email" required placeholder="Email" class="w-full rounded border px-3 py-2" />
    <input v-model="password" type="password" required placeholder="Password" class="w-full rounded border px-3 py-2" />
    <p v-if="error" class="text-sm text-red-600">{{ error }}</p>
    <button class="w-full rounded bg-orange-500 py-2 font-medium text-white">Login</button>
    <p class="text-sm">No account? <RouterLink to="/register" class="text-orange-600 underline">Register</RouterLink></p>
  </form>
</template>
