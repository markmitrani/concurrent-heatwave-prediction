git diff --name-only | while read file; do
  echo "Checking $file"
  if [ -f "$file" ]; then
    size=$(stat -c%s "$file")
    if [ "$size" -gt 89478485 ]; then
      echo "$file is $(du -h "$file" | cut -f1)"
    fi
    echo "done"
  else
    echo "$file is not a regular file or does not exist"
  fi
done
