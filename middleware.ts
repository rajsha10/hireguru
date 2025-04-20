import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"
import * as jose from 'jose'

// Define the shape of your JWT payload
interface JWTPayload {
  userId: string
  name: string
  role: 'candidate' | 'hr'
  iat: number
  exp: number
}

// Edge runtime compatible JWT verification
async function verifyJWTEdge(token: string): Promise<JWTPayload> {
  try {
    // Replace with your actual JWT secret
    const secret = new TextEncoder().encode(process.env.JWT_SECRET)
    
    console.log('Attempting to verify token with secret length:', secret.length)
    
    const { payload } = await jose.jwtVerify(token, secret)
    console.log('Token verified successfully:', payload)
    
    return payload as JWTPayload
  } catch (error) {
    console.error('JWT verification error:', error)
    console.error('Error name:', error.name)
    console.error('Error message:', error.message)
    throw error
  }
}

export async function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname
  console.log('Middleware called for path:', path)
  
  const isPublicPath =
    path === "/" ||
    path === "/login" ||
    path === "/signup" ||
    path === "/features" ||
    path === "/how-it-works" ||
    path === "/contact" ||
    path.startsWith("/api/auth")
    
  // If the path is public, allow access
  if (isPublicPath) {
    console.log('Allowing access to public path:', path)
    return NextResponse.next()
  }
  
  // Get token from cookies
  const token = request.cookies.get("token")?.value || ""
  console.log('Token from cookie:', token ? `${token.substring(0, 50)}...` : 'no token')
  
  // If no token and trying to access protected route, redirect to login
  if (!token) {
    console.log("No token found, redirecting to login")
    return NextResponse.redirect(new URL("/login", request.url))
  }
  
  try {
    // Verify the token using Edge-compatible method
    const decoded = await verifyJWTEdge(token)
    console.log('Decoded JWT:', {
      userId: decoded.userId,
      role: decoded.role,
      exp: decoded.exp,
      isExpired: decoded.exp ? decoded.exp * 1000 < Date.now() : 'unknown'
    })
    
    // Check role-based access
    if (path.startsWith("/hr-dashboard") && decoded.role !== "hr") {
      console.log('Non-HR user trying to access HR dashboard, redirecting to candidate dashboard')
      return NextResponse.redirect(new URL("/candidate-dashboard", request.url))
    }
    
    if (path.startsWith("/candidate-dashboard") && decoded.role !== "candidate") {
      console.log('Non-candidate user trying to access candidate dashboard, redirecting to HR dashboard')
      return NextResponse.redirect(new URL("/hr-dashboard", request.url))
    }
    
    // If everything is fine, allow the request
    console.log('Token valid, allowing access to:', path)
    return NextResponse.next()
  } catch (error) {
    console.error("Token verification failed:", error)
    
    // Clear the invalid token
    const response = NextResponse.redirect(new URL("/login", request.url))
    response.cookies.delete("token")
    
    return response
  }
}

// Configure the middleware to run on specific paths
export const config = {
  matcher: [
    '/candidate-dashboard/:path*',
    '/hr-dashboard/:path*',
    '/profile/:path*',
  ],
}