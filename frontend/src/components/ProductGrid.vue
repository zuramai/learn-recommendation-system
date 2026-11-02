<script setup lang="ts">
import { computed, watch } from 'vue'
import type { ProductCard as Card, RecCard } from '../api'
import { track } from '../composables/useTracker'
import ProductCard from './ProductCard.vue'

const props = defineProps<{ items: Card[]; surface: string; recs?: RecCard[] }>()
const emit = defineEmits<{ hide: [id: number] }>()

// book:start impressions
const seen = new Set<number>() // one impression per item id per grid instance
const idsKey = computed(() => props.items.map((i) => i.id).join(','))
watch(
  idsKey,
  () => {
    props.items.forEach((item, position) => {
      if (seen.has(item.id)) return
      seen.add(item.id)
      track({ item_id: item.id, event_type: 'impression', position, surface: props.surface })
    })
  },
  { immediate: true },
)
// book:end impressions

const recById = computed(() => new Map((props.recs ?? []).map((r) => [r.id, r])))
</script>

<template>
  <div class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
    <ProductCard
      v-for="(item, i) in items"
      :key="item.id"
      :item="item"
      :position="i"
      :surface="surface"
      :rec="recById.get(item.id)"
      @hide="emit('hide', $event)"
    />
  </div>
</template>
