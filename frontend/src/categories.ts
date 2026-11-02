import { ref } from 'vue'
import { api, type Category } from './api'

export const categories = ref<Category[]>([])
let loading: Promise<void> | null = null

export function loadCategories(): Promise<void> {
  if (!loading) {
    loading = api<Category[]>('/categories')
      .then((c) => {
        categories.value = c
      })
      .catch(() => {
        loading = null
      })
  }
  return loading
}

export function categoryParent(category_id: number): string {
  return categories.value.find((c) => c.id === category_id)?.parent_name ?? ''
}
