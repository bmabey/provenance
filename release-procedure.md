1.  Verify tests pass.

2.  Tag the commit

        git tag 1.2.3

3.  Push new version bump commit and tag to github

        git push trunk --tags

4.  Build source and wheel packages

        make dist

6.  Upload packages to PyPI

        make release
