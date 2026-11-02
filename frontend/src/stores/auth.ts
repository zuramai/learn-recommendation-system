import { defineStore } from 'pinia'
import { ApiError, api, post, type User } from '../api'

export const useAuth = defineStore('auth', {
  state: () => ({ user: null as User | null, loaded: false }),
  actions: {
    async fetchMe() {
      try {
        this.user = await api<User>('/auth/me')
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) this.user = null
        else throw e
      } finally {
        this.loaded = true
      }
    },
    async login(email: string, password: string) {
      this.user = await post<User>('/auth/login', { email, password })
    },
    async register(p: { email: string; password: string; name: string; city: string }) {
      this.user = await post<User>('/auth/register', p)
    },
    async logout() {
      await post('/auth/logout')
      this.user = null
    },
  },
})
