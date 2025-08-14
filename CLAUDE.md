# Claude Code Instructions

## Git Workflow

**IMPORTANT: Upon completion of each task, you MUST:**

1. **Add all changes to git staging**
   ```bash
   git add -A
   ```

2. **Create a descriptive commit**
   ```bash
   git commit -m "feat: [description of completed task]"
   ```
   
   Use conventional commit format:
   - `feat:` for new features
   - `fix:` for bug fixes  
   - `docs:` for documentation changes
   - `refactor:` for code refactoring
   - `test:` for test additions/changes
   - `chore:` for maintenance tasks

3. **Push to GitHub**
   ```bash
   git push origin main
   ```

### Example Workflow

After completing a task like "implement AI detection patterns":

```bash
git add -A
git commit -m "feat: implement data-driven AI detection patterns with 89% accuracy"
git push origin main
```

### Task Completion Checklist

- [ ] Task implementation complete
- [ ] Code tested and working
- [ ] Files saved
- [ ] Changes added to git (`git add -A`)
- [ ] Descriptive commit created (`git commit -m "..."`)
- [ ] Changes pushed to GitHub (`git push origin main`)

## Project-Specific Instructions

### AI Detector Chrome Extension

- Always test pattern changes before committing
- Update version numbers in manifest.json for significant changes
- Document accuracy improvements in commit messages
- Include performance metrics when relevant

### Code Quality

- Ensure no sensitive data (API keys, passwords) in commits
- Keep commits atomic and focused on single tasks
- Write clear, descriptive commit messages
- Push regularly to maintain backup and collaboration