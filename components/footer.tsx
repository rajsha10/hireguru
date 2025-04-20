export default function Footer() {
  return (
    <footer className="w-full border-t border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 py-6">
      <div className="container flex items-center justify-center text-center text-sm text-muted-foreground">
        &copy; {new Date().getFullYear()} InterviewIQ. All rights reserved.
      </div>
    </footer>
  )
}
