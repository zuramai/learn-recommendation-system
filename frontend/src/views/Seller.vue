<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ApiError, CITIES, api, del, formatIDR, patch, post, type SellerOrder, type SellerProduct, type Shop } from '../api'
import { categories } from '../categories'

const shop = ref<Shop | null>(null)
const loaded = ref(false)
const tab = ref<'products' | 'orders' | 'info'>('products')
const products = ref<SellerProduct[]>([])
const orders = ref<SellerOrder[]>([])
const error = ref('')

const shopForm = ref({ name: '', city: CITIES[0] as string })

interface Form {
  title: string
  description: string
  brand: string
  category_id: number
  price: number
  discount_pct: number // percent in the form, fraction on the wire
  free_shipping: boolean
  stock: number
  is_active: boolean
}
const blank = (): Form => ({
  title: '',
  description: '',
  brand: '',
  category_id: 0, // 0 = not chosen yet; the form refuses to submit until a category is picked
  price: 10000,
  discount_pct: 0,
  free_shipping: false,
  stock: 10,
  is_active: true,
})
const form = ref<Form>(blank())
const editingId = ref<number | null>(null)

async function loadShop() {
  try {
    shop.value = await api<Shop>('/seller/shop')
    await Promise.all([loadProducts(), loadOrders()])
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) shop.value = null
    else error.value = (e as Error).message
  } finally {
    loaded.value = true
  }
}
async function loadProducts() {
  products.value = await api<SellerProduct[]>('/seller/products')
}
async function loadOrders() {
  orders.value = await api<SellerOrder[]>('/seller/orders')
}
async function guarded(fn: () => Promise<unknown>) {
  error.value = ''
  try {
    await fn()
  } catch (e) {
    error.value = (e as Error).message
  }
}

const createShop = () =>
  guarded(async () => {
    shop.value = await post<Shop>('/seller/shop', shopForm.value)
    await Promise.all([loadProducts(), loadOrders()])
  })

function edit(p: SellerProduct) {
  editingId.value = p.id
  form.value = {
    title: p.title,
    description: p.description,
    brand: p.brand,
    category_id: p.category_id,
    price: p.price,
    discount_pct: Math.round(p.discount_pct * 100),
    free_shipping: p.free_shipping,
    stock: p.stock,
    is_active: p.is_active,
  }
}
function reset() {
  editingId.value = null
  form.value = blank()
}
const save = () =>
  guarded(async () => {
    if (!form.value.category_id) throw new Error('Choose a category')
    const body = { ...form.value, discount_pct: form.value.discount_pct / 100 }
    if (editingId.value === null) {
      const { is_active: _ignored, ...create } = body
      await post('/seller/products', create)
    } else {
      await patch(`/seller/products/${editingId.value}`, body)
    }
    reset()
    await loadProducts()
  })
const deactivate = (id: number) =>
  guarded(async () => {
    await del(`/seller/products/${id}`)
    await loadProducts()
  })
const setStatus = (id: number, status: string) =>
  guarded(async () => {
    await patch(`/seller/orders/${id}`, { status })
    await loadOrders()
  })

onMounted(async () => {
  await loadShop()
  reset()
})
</script>

<template>
  <h1 class="mb-3 text-xl font-semibold tracking-tight">Seller Center</h1>
  <p v-if="error" class="mb-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{{ error }}</p>

  <form v-if="loaded && !shop" class="max-w-sm space-y-3 card p-6" @submit.prevent="createShop">
    <h2 class="font-semibold">Open your shop</h2>
    <input v-model="shopForm.name" required placeholder="Shop name" class="input" />
    <select v-model="shopForm.city" class="input">
      <option v-for="c in CITIES" :key="c" :value="c">{{ c }}</option>
    </select>
    <button class="btn-primary w-full">Create shop</button>
  </form>

  <template v-if="shop">
    <div class="mb-3 flex gap-2 border-b">
      <button
        v-for="t in ['products', 'orders', 'info'] as const"
        :key="t"
        class="-mb-px border-b-2 px-3 py-2 text-sm capitalize"
        :class="tab === t ? 'border-primary-600 font-semibold text-primary-600' : 'border-transparent text-neutral-500'"
        @click="tab = t"
      >
        {{ t === 'info' ? 'Shop info' : t }}
      </button>
    </div>

    <div v-if="tab === 'products'" class="grid gap-4 lg:grid-cols-[1fr_320px]">
      <table class="h-fit w-full card overflow-hidden text-sm">
        <thead class="th">
          <tr><th class="p-2">Title</th><th>Price</th><th>Stock</th><th>Status</th><th /></tr>
        </thead>
        <tbody>
          <tr v-for="p in products" :key="p.id" class="border-t">
            <td class="p-2">{{ p.title }}</td>
            <td>{{ formatIDR(p.final_price) }}</td>
            <td>{{ p.stock }}</td>
            <td>{{ p.is_active ? 'active' : 'inactive' }}</td>
            <td class="space-x-2 whitespace-nowrap p-2 text-right">
              <button class="text-primary-600" @click="edit(p)">Edit</button>
              <button v-if="p.is_active" class="text-rose-500" @click="deactivate(p.id)">Deactivate</button>
            </td>
          </tr>
          <tr v-if="!products.length"><td colspan="5" class="p-4 text-center text-neutral-500">No products yet.</td></tr>
        </tbody>
      </table>
      <form class="space-y-2 card p-4 text-sm" @submit.prevent="save">
        <h2 class="font-semibold">{{ editingId === null ? 'New product' : `Edit product #${editingId}` }}</h2>
        <input v-model="form.title" required placeholder="Title" class="input" />
        <textarea v-model="form.description" placeholder="Description" rows="2" class="input" />
        <input v-model="form.brand" required placeholder="Brand" class="input" />
        <select v-model.number="form.category_id" class="input">
          <option :value="0" disabled>Select category…</option>
          <option v-for="c in categories" :key="c.id" :value="c.id">{{ c.parent_name }} › {{ c.name }}</option>
        </select>
        <label class="block">Price (IDR)
          <input v-model.number="form.price" type="number" min="1" class="input" /></label>
        <label class="block">Discount (%, 0–50)
          <input v-model.number="form.discount_pct" type="number" min="0" max="50" class="input" /></label>
        <label class="block">Stock
          <input v-model.number="form.stock" type="number" min="0" class="input" /></label>
        <label class="flex items-center gap-2"><input v-model="form.free_shipping" type="checkbox" /> Free shipping</label>
        <label v-if="editingId !== null" class="flex items-center gap-2"><input v-model="form.is_active" type="checkbox" /> Active</label>
        <div class="flex gap-2">
          <button class="btn-primary flex-1">{{ editingId === null ? 'Create' : 'Save' }}</button>
          <button v-if="editingId !== null" type="button" class="btn-secondary" @click="reset">Cancel</button>
        </div>
      </form>
    </div>

    <div v-else-if="tab === 'orders'" class="space-y-3">
      <p v-if="!orders.length" class="card p-8 text-center text-neutral-500">No orders yet.</p>
      <div v-for="o in orders" :key="o.id" class="card p-4">
        <div class="flex items-center justify-between">
          <div class="font-medium">Order #{{ o.id }} · {{ o.buyer_name }}</div>
          <span class="badge bg-neutral-100 text-neutral-700">{{ o.status }}</span>
        </div>
        <ul class="mt-1 text-sm">
          <li v-for="l in o.items" :key="l.item_id">{{ l.quantity }} × {{ l.title }} — {{ formatIDR(l.unit_price) }}</li>
        </ul>
        <div class="mt-2 flex items-center justify-between">
          <b class="text-sm">Total {{ formatIDR(o.total) }}</b>
          <button v-if="o.status === 'paid'" class="btn-primary !px-3 !py-1" @click="setStatus(o.id, 'shipped')">Mark shipped</button>
          <button v-if="o.status === 'shipped'" class="btn !px-3 !py-1 bg-emerald-600 text-white hover:bg-emerald-700" @click="setStatus(o.id, 'completed')">Mark completed</button>
        </div>
      </div>
    </div>

    <div v-else class="max-w-md card p-4 text-sm">
      <h2 class="text-lg font-semibold">{{ shop.name }}</h2>
      <p>City: {{ shop.city }}</p>
      <p>Rating: ⭐ {{ shop.shop_rating.toFixed(1) }}</p>
      <p>Mall: {{ shop.is_mall ? 'yes' : 'no' }}</p>
      <RouterLink :to="`/shops/${shop.id}`" class="mt-2 inline-block text-primary-600 hover:text-primary-700 hover:underline">View public shop page</RouterLink>
    </div>
  </template>
</template>
