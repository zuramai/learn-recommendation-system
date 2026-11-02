<script setup lang="ts">
import { ref } from 'vue'
import type { ProductCard, RecCard } from '../api'
import { formatIDR } from '../api'
import { track } from '../composables/useTracker'
import ProductImage from './ProductImage.vue'
import WhyThis from './WhyThis.vue'

const props = defineProps<{ item: ProductCard; position: number; surface: string; rec?: RecCard }>()
const emit = defineEmits<{ hide: [id: number] }>()
const showWhy = ref(false)

function notInterested() {
  track({ item_id: props.item.id, event_type: 'hide', position: props.position, surface: props.surface })
  emit('hide', props.item.id)
}
</script>

<template>
  <div class="flex flex-col rounded-lg bg-white p-2 shadow-sm transition hover:shadow-md">
    <RouterLink
      :to="{ path: `/products/${item.id}`, query: { src: surface, pos: String(position) } }"
      class="block"
    >
      <ProductImage :id="item.id" :brand="item.brand" :category-id="item.category_id" />
      <h3 class="mt-2 line-clamp-2 min-h-10 text-sm leading-5">{{ item.title }}</h3>
      <div class="mt-1 font-semibold text-orange-600">{{ formatIDR(item.final_price) }}</div>
      <div v-if="item.discount_pct > 0" class="text-xs text-gray-400">
        <span class="line-through">{{ formatIDR(item.price) }}</span>
        <span class="ml-1 rounded bg-red-100 px-1 text-red-600">-{{ Math.round(item.discount_pct * 100) }}%</span>
      </div>
      <div class="mt-1 flex items-center justify-between text-xs text-gray-500">
        <span>⭐ {{ item.rating_avg.toFixed(1) }} ({{ item.rating_count }})</span>
        <span>{{ item.shop_city }}</span>
      </div>
      <div v-if="!item.in_stock" class="mt-1 text-xs font-medium text-red-500">Out of stock</div>
    </RouterLink>
    <div v-if="rec" class="mt-2 flex items-center justify-between text-xs">
      <button class="text-orange-600 underline" @click="showWhy = !showWhy">Why this?</button>
      <button v-if="surface === 'home_feed'" class="text-gray-400 hover:text-red-500" @click="notInterested">
        Not interested
      </button>
    </div>
    <WhyThis v-if="rec && showWhy" :rec="rec" />
  </div>
</template>
