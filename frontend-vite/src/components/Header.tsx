import { Link } from "react-router-dom"
import { AlgoBrawlLogo } from "@/components/AlgoBrawlLogo"


export function Header() {
    return (
        <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex h-14 items-center justify-center">
                <Link to="/" className="flex items-center justify-center">
                    <AlgoBrawlLogo className="text-xl" />
                </Link>
            </div>
        </header>
    )
}
