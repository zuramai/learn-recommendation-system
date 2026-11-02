import { ref } from 'vue'
import { api, type FeatureInfo } from './api'

/** Feature descriptions from /api/recsys/info, cached for the "Why this?" panels. */
export const featureDescriptions = ref<Record<string, FeatureInfo>>({})
let loading: Promise<void> | null = null

export function loadFeatureDescriptions(): Promise<void> {
  if (!loading) {
    loading = api<{ features: FeatureInfo[] }>('/recsys/info')
      .then((info) => {
        featureDescriptions.value = Object.fromEntries(info.features.map((f) => [f.name, f]))
      })
      .catch(() => {
        loading = null
      })
  }
  return loading
}
