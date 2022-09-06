require 'nokogiri'
require 'open-uri'
require 'uri'

module Jekyll

  class RemoteContentTag < Liquid::Tag

    def initialize(tag_name, markup, tokens)
      #markup is what is defined in the tag. Lets make it a URL so devs
      #don't have to update code if the URL changes.
      url = markup

      #check if the URL is valid
      if url =~ URI::regexp
        #grab the remote document with nokogiri

        URI.open("https://google.com")

        doc = Nokogiri::HTML(URI.open(url))

        #search the document for the HTML element you want
        # @node = doc
        @node = doc.at_xpath("//div[@id='res_script']")
#         @node = doc.at_xpath("//html") do |config|
#   config.dtdload
# end

      else
        raise 'Invalid URL passed to RemoteContentTag'
      end

      super
    end

    def render(context)
      output = super
      if @node
        @node.to_s
      else
        "Something went wrong in RemoteContentTag"
      end
    end
  end
end

Liquid::Template.register_tag('remote_content', Jekyll::RemoteContentTag)
