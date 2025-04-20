import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'
import { verify } from 'jsonwebtoken'
import connectToDatabase from '@/lib/mongodb'
import User from '@/models/user'

export async function GET() {
  try {
    const cookieStore = cookies()
    const token = cookieStore.get('token')?.value

    if (!token) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    // Verify JWT token
    let decoded
    try {
      // Replace 'your_jwt_secret' with your actual JWT secret from environment variables
      decoded = verify(token, process.env.JWT_SECRET || 'your_jwt_secret')
    } catch (error) {
      return NextResponse.json(
        { error: 'Invalid token' },
        { status: 401 }
      )
    }

    // Connect to MongoDB
    await connectToDatabase()

    // Find the user by ID from token
    const userId = typeof decoded === 'object' && decoded !== null ? decoded.userId : null
    
    if (!userId) {
      return NextResponse.json(
        { error: 'Invalid token payload' },
        { status: 401 }
      )
    }

    const user = await User.findById(userId).select('-password')

    if (!user) {
      return NextResponse.json(
        { error: 'User not found' },
        { status: 404 }
      )
    }

    // Verify user is HR
    if (user.role !== 'hr') {
      return NextResponse.json(
        { error: 'Unauthorized, not an HR user' },
        { status: 403 }
      )
    }

    // Return HR profile data
    return NextResponse.json({
      id: user._id,
      name: user.name,
      email: user.email,
      company: user.company,
      role: user.role,
    })
  } catch (error) {
    console.error('Error in HR profile API:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}