import { defineStore } from 'pinia'
import { api, del, patch, post, type Cart } from '../api'
import { useAuth } from './auth'

export const useCart = defineStore('cart', {
  state: () => ({ cart: null as Cart | null }),
  getters: {
    count: (s) => s.cart?.lines.reduce((n, l) => n + l.quantity, 0) ?? 0,
  },
  actions: {
    async fetch() {
      if (!useAuth().user) {
        this.cart = null
        return
      }
      this.cart = await api<Cart>('/cart')
    },
    async add(item_id: number, quantity: number) {
      this.cart = await post<Cart>('/cart', { item_id, quantity })
    },
    async update(item_id: number, quantity: number) {
      this.cart = await patch<Cart>(`/cart/${item_id}`, { quantity })
    },
    async remove(item_id: number) {
      this.cart = await del<Cart>(`/cart/${item_id}`)
    },
  },
})
