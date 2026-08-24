#!/bin/bash
# deploy-web.sh — deploy website file(s) from the git repo to the live site.
#
# Usage (unchanged — single file, exactly as before):
#   ./deploy-web.sh index.html
#   ./deploy-web.sh index.html "updated hero section copy"
#
# Usage (new — batch, multiple files at once):
#   ./deploy-web.sh favicon.ico favicon-16x16.png favicon-32x32.png index.html -m "add favicon and logo schema"
#
# What it does:
#   1. git add + commit + push all given files (one commit covering all of them)
#   2. Copy each into /var/www/html (where nginx actually serves the live site from)
#   3. Verify the push landed on origin/main
#   4. Confirm the live site returns each file correctly
#
# Only works from inside /home/ubuntu/apfee (the repo root).

set -e  # stop immediately if any step fails, rather than continuing on a broken state

if [ "$#" -eq 0 ]; then
    echo "Usage:"
    echo "  ./deploy-web.sh <filename> [\"commit message\"]"
    echo "  ./deploy-web.sh <file1> <file2> ... -m \"commit message\""
    exit 1
fi

# ── Parse arguments: if -m is present, everything after it is the message and
# everything before it is the file list (any number of files). If -m is absent,
# fall back to the original two-positional-argument form (file, then message)
# so existing single-file usage keeps working exactly as before.
FILES=()
MSG=""
if [[ " $* " == *" -m "* ]]; then
    collecting_files=true
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "-m" ]; then
            collecting_files=false
            shift
            MSG="$1"
        elif [ "$collecting_files" = true ]; then
            FILES+=("$1")
        fi
        shift
    done
else
    FILES=("$1")
    MSG="${2:-Update $1}"
fi

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "Error: no files given."
    exit 1
fi

# Confirm every file actually exists before touching git at all
for f in "${FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "Error: $f not found in $(pwd)"
        echo "Run this from /home/ubuntu/apfee, and check the filename."
        exit 1
    fi
done

echo "=== 1. Committing and pushing ${#FILES[@]} file(s) ==="
for f in "${FILES[@]}"; do
    git add "$f"
done
if git diff --cached --quiet; then
    echo "No changes to commit — all given files are identical to the last commit. Skipping git steps."
else
    git commit -m "$MSG"
    git push origin main
fi

echo ""
echo "=== 2. Confirming the push landed on origin/main ==="
git log --oneline -1

echo ""
echo "=== 3. Deploying ${#FILES[@]} file(s) to the live webroot ==="
for f in "${FILES[@]}"; do
    sudo rm -f "/var/www/html/$f"
    sudo cp "/home/ubuntu/apfee/$f" "/var/www/html/$f"
    sudo chown www-data:www-data "/var/www/html/$f"
    echo "Copied to /var/www/html/$f"
done

echo ""
echo "=== 4. Verifying each file against the live domain ==="
ALL_OK=true
for f in "${FILES[@]}"; do
    if [ "$f" = "index.html" ]; then
        URL="https://tnltrader.com"
    elif [ "$f" = "privacy.html" ]; then
        URL="https://tnltrader.com/privacy"
    elif [ "$f" = "terms.html" ]; then
        URL="https://tnltrader.com/terms"
    else
        URL="https://tnltrader.com/$f"
    fi

    LOCAL_SIZE=$(wc -c < "$f")
    LIVE_SIZE=$(curl -sL "$URL" | wc -c)

    if [ "$LOCAL_SIZE" = "$LIVE_SIZE" ]; then
        echo "✅ $f — $URL matches ($LOCAL_SIZE bytes)"
    else
        echo "⚠️  $f — local $LOCAL_SIZE bytes vs live $LIVE_SIZE bytes at $URL"
        ALL_OK=false
    fi
done

echo ""
if [ "$ALL_OK" = true ]; then
    echo "✅ All ${#FILES[@]} file(s) deployed and verified."
else
    echo "⚠️  One or more files didn't match — try a hard-refresh (Cmd+Shift+R) before"
    echo "   assuming something's wrong; binary files like images/.ico won't always"
    echo "   compare cleanly through curl the way HTML does."
fi
