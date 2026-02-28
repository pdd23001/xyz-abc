import { Link, useLocation } from "react-router-dom"
import { cn } from "@/lib/utils"
import { Bot, BarChart2 } from "lucide-react"
import { BenchwarmerLogo } from "@/components/BenchwarmerLogo"


export function Header() {
    const location = useLocation()

    return (
        <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex h-14 items-center justify-between">
                {/* empty space for centering */}
                <div className="flex-1" />
                <Link to="/" className="flex items-center justify-center">
                    <BenchwarmerLogo className="text-xl" />
                </Link>
                <nav className="flex-1 flex justify-end items-center space-x-6 text-sm font-medium">
                    <Link
                        to="/"
                        className={cn(
                            "transition-colors hover:text-foreground/80 flex items-center gap-2",
                            location.pathname === "/" ? "text-foreground" : "text-foreground/60"
                        )}
                    >
                        <Bot className="h-4 w-4" />
                        Chat
                    </Link>
                    <Link
                        to="/benchmarks"
                        className={cn(
                            "transition-colors hover:text-foreground/80 flex items-center gap-2",
                            location.pathname === "/benchmarks" ? "text-foreground" : "text-foreground/60"
                        )}
                    >
                        <BarChart2 className="h-4 w-4" />
                        Benchmarks
                    </Link>
                </nav>
            </div>
        </header>
    )
}
