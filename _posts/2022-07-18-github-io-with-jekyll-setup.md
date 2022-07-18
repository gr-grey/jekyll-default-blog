---
layout: "post"
title: "Blogging with Jekyll and hosting it on github"
date: 2022-07-18 15:49:00
categories: coding tools
---


Jekyll is a static site generator. It lets you to write your posts as markdown files (instead of html, yuk!)
and takes care of all the other site hosting bussiness.
It creates this site that you're currently viewing!

### Creating a blogging site and host it on github in 3 simple steps.

1. Install Jekyll.

    Follow the installation guides from the Jekyll website [here](https://jekyllrb.com/docs/installation/).

1. Site initialization.
    - Once Jekyll is installed. You can create a template site by running the command `jekyll new site_name` from your terminal.
    It creates a folder with your chosen 'site_name' with basic files needed for the site.

1. Now if you put the repo on github with the name 'your-git-name.github.io', the site will show up at https://your-git-name.github.io/
    - To do that, first go to github.com, create a new empty repository with the name 'your-git-name.github.io' where your-git-name is your github account name,
    (for me it's gr-grey.github.io), do not initialize README, license or gitignore files to avoid error, you can add them later.
    - In your local site folder, initialize the repo as a git repository by running the `git init -b main` command from terminal.
    Then stage and commit by `git add . && git commit -m "initial commit"`
    Set the remote URL by `git remote add origin 'https://github/your-git-name/your-git-name.github.io'`
    (you can copy the URL on github's quick setup page when you create the repo),
    and push the local files by running `git push -u origin main`
    
You should see a nice looking page with a post 'Welcome to Jekyll!'.
Posts exist as indvidual markdown files in the `_posts` folder.
You can find the file for this post, play with it, add more posts and modify other settings such as site name and about page.

---

#### Some notes:

1. After you push the repository to github, you may need to wait a minute or two to be able to visit the site from your browser, 
as github needs to complie the website before serving them.
1. 'your-git-name.github.io' is a special repository name recognized by github.
Github will automatically compile and host your website with the content in the 'your-git-name.github.io' repo.
