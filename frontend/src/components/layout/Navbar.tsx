export default function Navbar() {
  return (
    <nav className="w-full border-b bg-white">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-4">

        <div className="text-xl font-semibold">
          NFC
        </div>
        
        <ul className="flex items-center gap-8 text-sm">
          <li><a href="/" className="hover:text-blue-600">Home</a></li>
          <li><a href="/compare" className="hover:text-blue-600">Compare Articles</a></li>
          <li><a href="/about" className="hover:text-blue-600">About</a></li>
          <li><a href="/settings" className="hover:text-blue-600">Settings</a></li>
        </ul>
      </div>
    </nav>
  );
}
