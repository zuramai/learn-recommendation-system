export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function api<T = void>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch('/api' + path, {
    credentials: 'same-origin',
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
  })
  if (!res.ok) {
    let msg = res.statusText
    try {
      const body = await res.json()
      if (body?.error) msg = body.error
    } catch {
      /* no json body */
    }
    throw new ApiError(res.status, msg)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export function post<T = void>(path: string, body?: unknown): Promise<T> {
  return api<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })
}
export function patch<T = void>(path: string, body: unknown): Promise<T> {
  return api<T>(path, { method: 'PATCH', body: JSON.stringify(body) })
}
export function del<T = void>(path: string): Promise<T> {
  return api<T>(path, { method: 'DELETE' })
}

export const CITIES = ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Makassar']

export interface User {
  id: number
  email: string
  name: string
  city: string
  is_admin: boolean
  shop_id: number | null
}

export interface ProductCard {
  id: number
  title: string
  brand: string
  price: number
  discount_pct: number
  final_price: number
  free_shipping: boolean
  rating_avg: number
  rating_count: number
  category_id: number
  category_name: string
  shop_id: number
  shop_name: string
  shop_city: string
  in_stock: boolean
}

export interface ShopSummary {
  id: number
  name: string
  city: string
  is_mall: boolean
  shop_rating: number
}

export interface ProductDetail extends ProductCard {
  description: string
  stock: number
  shop: ShopSummary
}

export interface Contribution {
  feature: string
  value: number
  contribution: number
}

export interface RecCard extends ProductCard {
  score: number
  source: string
  rank: number
  contributions: Contribution[]
}

export interface RecsResponse {
  items: RecCard[]
  model_version?: string
  latency_ms?: number
  candidates_considered?: number
  fallback: boolean
}

export interface ProductList {
  items: ProductCard[]
  total: number
  page: number
  page_size: number
}

export interface Category {
  id: number
  name: string
  parent_name: string
}

export interface CartLine {
  item: ProductCard
  quantity: number
  line_total: number
}
export interface Cart {
  lines: CartLine[]
  subtotal: number
  shipping_fee: number
  total: number
}

export interface OrderLine {
  item_id: number
  title: string
  quantity: number
  unit_price: number
}
export interface Order {
  id: number
  shop_id: number
  shop_name: string
  status: 'paid' | 'shipped' | 'completed'
  subtotal: number
  shipping_fee: number
  total: number
  created_at: string
  items: OrderLine[]
}
export interface SellerOrder extends Omit<Order, 'shop_name'> {
  buyer_name: string
}

export interface Shop {
  id: number
  name: string
  city: string
  is_mall: boolean
  shop_rating: number
  product_count?: number
}

export interface SellerProduct extends ProductCard {
  stock: number
  is_active: boolean
  description: string
}

export interface FeatureInfo {
  name: string
  group: string
  description: string
  gain_share: number
}
export interface ModelInfo {
  model_version: string
  trained_at: string
  metrics: Record<string, number>
  features: FeatureInfo[]
  events_loaded: number
  train_rows: number
}

export interface MyEvent {
  id: number
  ts: string
  event_type: string
  item_id: number
  item_title: string
  surface: string | null
  position: number | null
  dwell_sec: number | null
  price_paid: number | null
}
export interface MyEvents {
  events: MyEvent[]
  searches: { ts: string; query: string }[]
}

export const formatIDR = (n: number) => 'Rp' + n.toLocaleString('id-ID')

export const RECSYS_DOWN_MSG = 'Recommender service is not running (cd recommender && uv run recsys serve)'
