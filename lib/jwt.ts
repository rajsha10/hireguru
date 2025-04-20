import jwt from "jsonwebtoken"
import { cookies } from "next/headers"
import type { NextRequest } from "next/server"

const JWT_SECRET = process.env.JWT_SECRET
const JWT_LIFETIME = process.env.JWT_LIFETIME

export function createJWT(payload: any) {
  return jwt.sign(payload, JWT_SECRET, {
    expiresIn: '1d',
  })
}

export function verifyJWT(token: string) {
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    if (!decoded) throw new Error('Token verification failed');
    return decoded;
  } catch (error) {
    console.error('JWT verification error:', error.message);
    throw error; // Re-throw to handle in middleware
  }
}

export async function setTokenCookie(token: string) {
  const cookieStore = await cookies()
  await cookieStore.set("token", token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "strict",
    maxAge: 24 * 60 * 60,
    path: "/",
  })
}

export async function getTokenFromCookies() {
  const cookieStore = await cookies()
  return cookieStore.get("token")?.value
}

export function getTokenFromRequest(request: NextRequest) {
  const token = request.cookies.get("token")?.value
  return token
}

export async function clearTokenCookie() {
  const cookieStore = await cookies()
  await cookieStore.delete("token")
}