import { Skeleton } from "@/components/ui/skeleton"

export default function Loading() {
  return (
    <div className="container py-10">
      <div className="space-y-6 max-w-2xl mx-auto">
        <Skeleton className="h-12 w-[250px]" />
        <Skeleton className="h-[400px] rounded-xl" />
      </div>
    </div>
  )
}
