<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { CITIES } from '../api'
import { useAuth } from '../stores/auth'

const auth = useAuth()
const route = useRoute()
const router = useRouter()
const form = ref({ email: '', password: '', name: '', city: CITIES[0] as string })
const error = ref('')

async function submit() {
  error.value = ''
  try {
    await auth.register(form.value)
    void router.push(typeof route.query.next === 'string' ? route.query.next : '/')
  } catch (e) {
    error.value = (e as Error).message
  }
}
</script>

<template>
  <form class="mx-auto max-w-sm space-y-3 card p-6" @submit.prevent="submit">
    <h1 class="text-xl font-semibold tracking-tight">Register</h1>
    <input v-model="form.name" required placeholder="Name" class="input" />
    <input v-model="form.email" type="email" required placeholder="Email" class="input" />
    <input v-model="form.password" type="password" required minlength="8" placeholder="Password (min 8 chars)" class="input" />
    <select v-model="form.city" class="input">
      <option v-for="c in CITIES" :key="c" :value="c">{{ c }}</option>
    </select>
    <p v-if="error" class="text-sm text-rose-600">{{ error }}</p>
    <button class="btn-primary w-full">Create account</button>
    <p class="text-sm">Have an account? <RouterLink to="/login" class="text-primary-600 hover:text-primary-700 hover:underline">Login</RouterLink></p>
  </form>
</template>
