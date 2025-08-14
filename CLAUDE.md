# Claude Code Instructions

## 🎯 Implementation Roadmap

**CRITICAL: Follow the comprehensive implementation roadmap for all development work**

📍 **Roadmap Location**: `C:\Users\Kevin\ai_detector\IMPLEMENTATION_ROADMAP.md`

### Roadmap Sections:
1. **File & Directory Organization** - Optimal project structure
2. **Code Architecture** - Design patterns and best practices  
3. **Consolidation** - Removing duplication and redundancy
4. **Integration** - Module communication and APIs
5. **Test Coverage** - Unit, integration, E2E, and performance tests

### Before Starting Any Task:
1. Review the relevant section in IMPLEMENTATION_ROADMAP.md
2. Check dependencies and prerequisites
3. Follow the defined patterns and standards
4. Update progress in the roadmap

## Git Workflow

**IMPORTANT: Upon completion of each task OR each phase, you MUST:**

1. **Add all changes to git staging**
   ```bash
   git add -A
   ```

2. **Create a descriptive commit**
   ```bash
   git commit -m "feat: [description of completed task/phase]"
   ```
   
   Use conventional commit format:
   - `feat:` for new features
   - `fix:` for bug fixes  
   - `docs:` for documentation changes
   - `refactor:` for code refactoring
   - `test:` for test additions/changes
   - `chore:` for maintenance tasks
   - `phase:` for completed implementation phases

3. **Push to GitHub** (if remote is configured)
   ```bash
   git push origin main
   ```
   
   **Note:** If no remote is configured, inform the user to:
   1. Create a repository on GitHub
   2. Add the remote: `git remote add origin https://github.com/username/repo-name.git`
   3. Push: `git push -u origin main`

## Phase Management

**CONTINUOUS IMPLEMENTATION: When a phase completes, automatically proceed to the next phase without prompting. Execute the entire implementation roadmap/todo list straight through.**

### Phase Completion Actions:
1. **Commit and push all phase changes**
2. **Update phase status in IMPLEMENTATION_ROADMAP.md**  
3. **Automatically begin next phase immediately**
4. **Continue until entire roadmap is complete**

### No Prompting Between Phases:
- Do NOT ask for permission to continue to next phase
- Do NOT wait for user confirmation between phases
- Execute the full roadmap continuously and systematically
- Only stop if critical errors occur or roadmap is complete

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