import { useState } from "react";
import { Menu, Home, Newspaper, Settings, Info } from "lucide-react";
import SidebarItem from "./SidebarItem";

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={`h-screen border-r bg-white transition-all duration-300 
      ${collapsed ? "w-20" : "w-64"}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-4 border-b">
        {!collapsed && <h1 className="text-xl font-semibold">NFC</h1>}

        <button
          className="p-2 rounded hover:bg-gray-100"
          onClick={() => setCollapsed(!collapsed)}
        >
          <Menu size={20} />
        </button>
      </div>

      {/* Links */}
      <div className="mt-4 flex flex-col gap-2">
        <SidebarItem
          icon={<Home size={20} />}
          label="Home"
          collapsed={collapsed}
          href="/"
        />

        <SidebarItem
          icon={<Newspaper size={20} />}
          label="Compare Articles"
          collapsed={collapsed}
          href="/compare"
        />

        <SidebarItem
          icon={<Info size={20} />}
          label="About"
          collapsed={collapsed}
          href="/about"
        />

        <SidebarItem
          icon={<Settings size={20} />}
          label="Settings"
          collapsed={collapsed}
          href="/settings"
        />
      </div>
    </aside>
  );
}
