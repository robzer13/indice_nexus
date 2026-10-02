export default function OroTitanLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="relative left-1/2 w-[min(1600px,calc(100vw-24px))] -translate-x-1/2 sm:w-[min(1600px,calc(100vw-48px))]">
      {children}
    </div>
  );
}
