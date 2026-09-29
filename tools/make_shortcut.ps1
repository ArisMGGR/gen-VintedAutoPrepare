# Creates VintedAutoPrepare.lnk in the project folder (called by Install.bat).
# Shortcuts store absolute paths, so this runs on your PC rather than being committed.
$root = Split-Path -Parent $PSScriptRoot
$lnkPath = Join-Path $root 'VintedAutoPrepare.lnk'
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($lnkPath)
$shortcut.TargetPath = Join-Path $root 'venv\Scripts\pythonw.exe'  # pythonw: no console window
$shortcut.Arguments = '"' + (Join-Path $root 'run.py') + '"'
$shortcut.WorkingDirectory = $root
$shortcut.IconLocation = Join-Path $root 'assets\icon.ico'
$shortcut.Description = 'Prepare photos and listings for Vinted'
$shortcut.Save()

# Give the shortcut the same AppUserModelID the app sets at startup (vinted_prep/__init__.py),
# so Windows treats the shortcut and the running window as one app: our icon on the taskbar,
# and pinning the taskbar button pins this shortcut rather than python.
$appId = 'ArisMGGR.VintedAutoPrepare'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class ShortcutAppId {
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    struct PropertyKey { public Guid fmtid; public uint pid; }

    [StructLayout(LayoutKind.Explicit)]
    struct PropVariant {
        [FieldOffset(0)] public ushort vt;
        [FieldOffset(8)] public IntPtr ptr;
    }

    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPropertyStore {
        void GetCount(out uint count);
        void GetAt(uint index, out PropertyKey key);
        void GetValue(ref PropertyKey key, out PropVariant value);
        void SetValue(ref PropertyKey key, ref PropVariant value);
        void Commit();
    }

    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    static extern void SHGetPropertyStoreFromParsingName(
        string path, IntPtr bindCtx, int flags, ref Guid riid, out IPropertyStore store);

    public static void Set(string lnkPath, string appId) {
        Guid iid = typeof(IPropertyStore).GUID;
        IPropertyStore store;
        SHGetPropertyStoreFromParsingName(lnkPath, IntPtr.Zero, 2 /* GPS_READWRITE */, ref iid, out store);
        var key = new PropertyKey { fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), pid = 5 };
        var value = new PropVariant { vt = 31 /* VT_LPWSTR */, ptr = Marshal.StringToCoTaskMemUni(appId) };
        try {
            store.SetValue(ref key, ref value);
            store.Commit();
        } finally {
            Marshal.FreeCoTaskMem(value.ptr);
            Marshal.ReleaseComObject(store);
        }
    }
}
'@
try {
    [ShortcutAppId]::Set($lnkPath, $appId)
} catch {
    # Not fatal: the shortcut still works, the taskbar may just show python's icon.
    Write-Warning "Could not set the shortcut's taskbar ID: $_"
}
