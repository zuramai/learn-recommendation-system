import { createRouter, createWebHistory } from 'vue-router'
import { useAuth } from './stores/auth'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: () => import('./views/Home.vue') },
    { path: '/search', component: () => import('./views/Search.vue') },
    { path: '/products/:id', component: () => import('./views/Product.vue') },
    { path: '/shops/:id', component: () => import('./views/Shop.vue') },
    { path: '/cart', component: () => import('./views/Cart.vue'), meta: { requiresAuth: true } },
    { path: '/checkout', component: () => import('./views/Checkout.vue'), meta: { requiresAuth: true } },
    { path: '/orders', component: () => import('./views/Orders.vue'), meta: { requiresAuth: true } },
    { path: '/login', component: () => import('./views/Login.vue') },
    { path: '/register', component: () => import('./views/Register.vue') },
    { path: '/seller', component: () => import('./views/Seller.vue'), meta: { requiresAuth: true } },
    { path: '/how-it-works', component: () => import('./views/HowItWorks.vue') },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

router.beforeEach(async (to) => {
  const auth = useAuth()
  if (!auth.loaded) await auth.fetchMe()
  if (to.meta.requiresAuth && !auth.user) {
    return { path: '/login', query: { next: to.fullPath } }
  }
})

export default router
