#!/bin/bash
# Fix GStreamer plugin registration for my_tile_batcher

set -e

echo "=== Fixing GStreamer Plugin Registration ==="

# 1. Rebuild tile batcher plugin
echo ""
echo "Step 1: Rebuilding tile batcher plugin..."
cd /home/user/ds_pipeline/my_tile_batcher/src
make clean
make

# 2. Install plugin to user directory
echo ""
echo "Step 2: Installing plugin..."
mkdir -p ~/.local/share/gstreamer-1.0/plugins/
cp libnvtilebatcher.so ~/.local/share/gstreamer-1.0/plugins/

# 3. Clear GStreamer plugin cache
echo ""
echo "Step 3: Clearing GStreamer plugin cache..."
rm -rf ~/.cache/gstreamer-1.0/
mkdir -p ~/.cache/gstreamer-1.0/

# 4. Re-scan plugins
echo ""
echo "Step 4: Re-scanning plugins..."
export GST_PLUGIN_PATH=$HOME/.local/share/gstreamer-1.0/plugins:/home/user/ds_pipeline/my_steach
gst-inspect-1.0 --gst-plugin-path=$GST_PLUGIN_PATH nvtilebatcher > /dev/null 2>&1 || true

# 5. Verify property exists
echo ""
echo "Step 5: Verifying tile-offset-y property..."
gst-inspect-1.0 --gst-plugin-path=$GST_PLUGIN_PATH nvtilebatcher | grep -A2 "tile-offset-y"

echo ""
echo "✅ Plugin registration complete!"
echo ""
echo "Properties available:"
gst-inspect-1.0 --gst-plugin-path=$GST_PLUGIN_PATH nvtilebatcher | grep -E "^\s+(gpu-id|panorama-width|panorama-height|tile-offset-y|silent)"
