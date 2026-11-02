<script setup lang="ts">
import { computed } from 'vue'
import { categoryParent } from '../categories'

const props = defineProps<{ id: number; brand: string; categoryId: number }>()

const EMOJI: Record<string, string> = {
  Electronics: '📱',
  Fashion: '👕',
  Beauty: '💄',
  'Home & Living': '🏠',
  'Mother & Baby': '🍼',
  Groceries: '🍜',
  Sports: '🏋️',
  Pets: '🐾',
}
const emoji = computed(() => EMOJI[categoryParent(props.categoryId)] ?? '🛍️')
const bg = computed(() => `hsl(${(props.id * 47) % 360} 70% 85%)`)
</script>

<template>
  <div
    class="flex aspect-square w-full flex-col items-center justify-center gap-1 overflow-hidden rounded"
    :style="{ background: bg }"
  >
    <span class="text-5xl">{{ emoji }}</span>
    <span class="max-w-full truncate px-2 text-xs font-semibold text-gray-700">{{ brand }}</span>
  </div>
</template>
