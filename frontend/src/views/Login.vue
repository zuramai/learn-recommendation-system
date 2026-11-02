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
  <form class="mx-auto max-w-sm space-y-3 card p-6" @submit.prevent="submit">
    <h1 class="text-xl font-semibold tracking-tight">Login</h1>
    <div class="rounded bg-primary-50 p-2 text-xs text-primary-800">
      Demo: user43@demo.test / password, seller1@demo.test / password, admin@demo.test / password
    </div>
    <input v-model="email" type="email" required placeholder="Email" class="input" />
    <input v-model="password" type="password" required placeholder="Password" class="input" />
    <p v-if="error" class="text-sm text-rose-600">{{ error }}</p>
    <button class="btn-primary w-full">Login</button>
    <p class="text-sm">No account? <RouterLink to="/register" class="text-primary-600 hover:text-primary-700 hover:underline">Register</RouterLink></p>
  </form>
</template>
