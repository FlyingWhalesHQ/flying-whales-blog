module Jekyll

  module CategoryArchiveUtil
    def self.archive_base(site)
      site.config['category_archive'] && site.config['category_archive']['path'] || ''
    end
  end

  # Generator class invoked from Jekyll
  class CategoryArchiveGenerator < Generator
    def generate(site)
      posts_group_by_category(site).each do |category, list|
        site.pages << CategoryArchivePage.new(site, CategoryArchiveUtil.archive_base(site), category, list)
      end
    end

    def posts_group_by_category(site)
      category_map = {}
      site.posts.each {|p| p.categories.each {|c| (category_map[c] ||= []) << p } }
      category_map
    end
  end

  # Tag for generating a link to a category archive page
  class CategoryArchiveLinkTag < Liquid::Block

    def initialize(tag_name, category, tokens)
      super
      @category = category.split(' ').first || category
    end

    def render(context)
      # If the category is a variable in the current context, expand it
      if context.has_key?(@category)
	      category = context[@category]
      else
	      category = @category
      end


      if context.registers[:site].config['category_archive'] && context.registers[:site].config['category_archive']['slugify']
        category = Utils.slugify(category)
      end

      href = File.join('/', context.registers[:site].baseurl, context.environments.first['site']['category_archive']['path'],
                       category, 'index.html')
      "<a href=\"#{href}\">#{super}</a>"
    end
  end

  # Actual page instances
  class CategoryArchivePage < Page
    ATTRIBUTES_FOR_LIQUID = %w[
      category,
      content
    ]

    def initialize(site, dir, category, posts)
      @site = site
      @dir = dir
      @category = category

      if site.config['category_archive'] && site.config['category_archive']['slugify']
        @category_dir_name = Utils.slugify(@category) # require sanitize here
      else
        @category_dir_name = @category
      end

      @layout =  site.config['category_archive'] && site.config['category_archive']['layout'] || 'category_archive'
      self.ext = '.html'
      self.basename = 'index'
      self.content = <<-EOS
{% for post in page.posts %}
<li>
  <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
  <h3>
    <a class="post-link" href="{{ post.url | relative_url }}">
      {{ post.title | escape }}
    </a>
  </h3>
  {%- if site.show_excerpts -%}
    {{ post.excerpt }}
  {%- endif -%}
</li>
{% endfor %}
      EOS
      self.data = {
          'layout' => @layout,
          'type' => 'archive',
          'title' => "#{@category}",
          'path' => File.join('/',@category_dir_name, 'index.html'),
          'posts' => posts,
          'url' => File.join('/',
                     CategoryArchiveUtil.archive_base(site),
                     @category_dir_name, 'index.html')
      }
    end

    def render(layouts, site_payload)
      payload = {
          'page' => self.to_liquid,
          'paginator' => pager.to_liquid
      }.merge(site_payload)
      do_layout(payload, layouts)
    end

    def to_liquid(attr = nil)
      self.data.merge({
                               'content' => self.content,
                               'category' => @category
                           })
    end

    def destination(dest)
      File.join('/', dest, @dir, @category_dir_name, 'index.html')
    end

  end
end

Liquid::Template.register_tag('categorylink', Jekyll::CategoryArchiveLinkTag)
