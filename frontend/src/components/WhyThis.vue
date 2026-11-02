<script setup lang="ts">
import { computed, onMounted } from 'vue'
import type { RecCard } from '../api'
import { featureDescriptions, loadFeatureDescriptions } from '../recsysInfo'

const props = defineProps<{ rec: RecCard }>()

// Same wording as pipeline.SOURCES in the recommender.
const SOURCES: Record<string, string> = {
  recent: 'you viewed it',
  covisit: 'viewed together with your items',
  category_popular: 'popular in a category you like',
  trending: 'trending now',
  same_category: 'popular in the same category',
}

onMounted(() => void loadFeatureDescriptions())

const maxAbs = computed(() => Math.max(1e-9, ...props.rec.contributions.map((c) => Math.abs(c.contribution))))
const describe = (f: string) => featureDescriptions.value[f]?.description ?? f
</script>

<template>
  <div class="mt-2 rounded-lg border border-neutral-200 bg-neutral-50 p-3 text-xs">
    <p>
      <span class="font-semibold">Because:</span> {{ SOURCES[rec.source] ?? rec.source }}
    </p>
    <p class="text-neutral-500">score {{ rec.score.toFixed(3) }} · rank #{{ rec.rank }}</p>
    <ul v-if="rec.contributions.length" class="mt-1 space-y-1.5">
      <li v-for="c in rec.contributions" :key="c.feature">
        <div class="leading-tight text-neutral-700" :title="c.feature">{{ describe(c.feature) }}</div>
        <div class="flex items-center gap-2">
          <span class="w-12 shrink-0 text-neutral-500">{{ c.value.toFixed(2) }}</span>
          <div class="h-2 flex-1 rounded bg-neutral-200">
            <div
              class="h-2 rounded"
              :class="c.contribution >= 0 ? 'bg-emerald-400' : 'bg-rose-400'"
              :style="{ width: (Math.abs(c.contribution) / maxAbs) * 100 + '%' }"
            />
          </div>
          <span class="w-12 shrink-0 text-right" :class="c.contribution >= 0 ? 'text-emerald-700' : 'text-rose-700'">
            {{ c.contribution >= 0 ? '+' : '−' }}{{ Math.abs(c.contribution).toFixed(2) }}
          </span>
        </div>
      </li>
    </ul>
  </div>
</template>
