package de.mpii.mr.utils;

import org.apache.commons.lang3.StringUtils;
import org.jsoup.nodes.Document;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import org.apache.log4j.Logger;
import org.jsoup.nodes.Element;
import org.tartarus.snowball.ext.PorterStemmer;

/**
 *
 * @author khui
 */
public class CleanContentTxt  implements java.io.Serializable{

    private static final Logger logger = Logger.getLogger(CleanContentTxt.class);

    private static final PorterStemmer porterStemmer = new PorterStemmer();

    private final StanfordCoreNLP pipeline;

    private int MAX_TOKEN_LENGTH = 64;       // Throw away tokens longer than this.

    public CleanContentTxt(int ignoreTermLength) {
        // initialize Stanford NLP pipeline
        Properties properties = new Properties();
        properties.put("annotators", "tokenize, ssplit");
        properties.put("encoding", "UTF-8");
        properties.put("parse.maxlen", "20");
        properties.put("tokenize.options", "untokenizable=noneDelete");
        pipeline = new StanfordCoreNLP(properties);
        this.MAX_TOKEN_LENGTH = ignoreTermLength;
    }

    public static List<String> queryTxt(String querystr) {
        // replace left-over tags, to generate the same term 
        // with the document content
        String cleantxt = querystr.replaceAll("<.*?[/]{0,1}>", "").toLowerCase();
        cleantxt = cleantxt.replaceAll("'s", "");
        List<String> termlist = new ArrayList();
        String[] terms = cleantxt.split(" ");
        if (terms.length > 0) {
            termlist = Arrays.asList(terms);
        }
        return termlist;
    }

    private List<String> clean2sentences(String rawTxt, boolean removeStopWords) {
        List<String> indexContentBuilder = new ArrayList<>();
        StringBuilder singleSentence;
        try {
            // annotate contents using Stanford CoreNLP
            Annotation annotation = new Annotation(rawTxt);
            pipeline.annotate(annotation);
            // clean up textual contents
            for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
                singleSentence = new StringBuilder();
                for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                    String word = token.get(CoreAnnotations.TextAnnotation.class).toLowerCase();

                    // replace "." if word doesn't end with "."
                    if (!word.endsWith(".")) {
                        word = word.replaceAll("\\.", " ");
                    }

                    // replace characters that are neither letter nor digit
                    char[] chars = word.toCharArray();
                    for (int i = 0; i < chars.length; i++) {
                        if (!Character.isLetterOrDigit(chars[i]) && chars[i] != '.' && chars[i] != '\'') {
                            chars[i] = ' ';
                        }
                    }
                    word = new String(chars);

                    // tokenize into subwords
                    for (String subword : word.split(" ")) {

                        // ignore long subwords
                        if (subword.length() > MAX_TOKEN_LENGTH && subword.length() < 2) {
                            continue;
                        }

                        // ignore subwords not containing a letter or symbol                                
                        boolean ignore = true;
                        for (int i = 0; i < subword.length() && ignore; i++) {
                            ignore = !Character.isLetterOrDigit(subword.charAt(i));
                        }
                        if (ignore) {
                            continue;
                        }

                        // ignore subwords containing more than four digits
                        int digitCount = 0;
                        for (int i = 0; i < subword.length(); i++) {
                            if (Character.isDigit(subword.charAt(i))) {
                                digitCount++;
                            } else {
                                digitCount = 0;
                            }
                        }
                        if (digitCount > 4) {
                            continue;
                        }

                        // ignore subwords containing more than three identical consecutive letters
                        int letterCount = 1;
                        for (int i = 0; i < subword.length(); i++) {
                            if (Character.isLetter(subword.charAt(i)) && (i > 0 && subword.charAt(i) == subword.charAt(i - 1))) {
                                letterCount++;
                            } else {
                                letterCount = 1;
                            }
                        }
                        if (letterCount > 3) {
                            continue;
                        }

                        if (removeStopWords) {
                            if (StopWordsFilter.isStopWordOrNumber(subword)) {
                                continue;
                            }
                        }
                        singleSentence.append(subword).append(" ");

                    }
                }
                indexContentBuilder.add(singleSentence.toString().trim());
            }
        } catch (Exception ex) {
            logger.error("", ex);
        }
        return indexContentBuilder;
    }

    public List<String> clean2tokens(String rawTxt, boolean removeStopWords) {
        List<String> indexContentBuilder = new ArrayList<>();
        try {
            // annotate contents using Stanford CoreNLP
            Annotation annotation = new Annotation(rawTxt);

            pipeline.annotate(annotation);

            // clean up textual contents
            for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
                for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                    String word = token.get(CoreAnnotations.TextAnnotation.class).toLowerCase();

                    // replace "." if word doesn't end with "."
                    if (!word.endsWith(".")) {
                        word = word.replaceAll("\\.", " ");
                    }

                    // replace characters that are neither letter nor digit
                    char[] chars = word.toCharArray();
                    for (int i = 0; i < chars.length; i++) {
                        if (!Character.isLetterOrDigit(chars[i]) && chars[i] != '.' && chars[i] != '\'') {
                            chars[i] = ' ';
                        }
                    }
                    word = new String(chars);

                    // tokenize into subwords
                    for (String subword : word.split(" ")) {

                        // ignore long subwords
                        if (subword.length() > MAX_TOKEN_LENGTH && subword.length() < 2) {
                            continue;
                        }

                        // ignore subwords not containing a letter or symbol
                        boolean ignore = true;
                        for (int i = 0; i < subword.length() && ignore; i++) {
                            ignore = !Character.isLetterOrDigit(subword.charAt(i));
                        }
                        if (ignore) {
                            continue;
                        }

                        // ignore subwords containing more than four digits
                        int digitCount = 0;
                        for (int i = 0; i < subword.length(); i++) {
                            if (Character.isDigit(subword.charAt(i))) {
                                digitCount++;
                            } else {
                                digitCount = 0;
                            }
                        }
                        if (digitCount > 4) {
                            continue;
                        }

                        // ignore subwords containing more than three identical consecutive letters
                        int letterCount = 1;
                        for (int i = 0; i < subword.length(); i++) {
                            if (Character.isLetter(subword.charAt(i)) && (i > 0 && subword.charAt(i) == subword.charAt(i - 1))) {
                                letterCount++;
                            } else {
                                letterCount = 1;
                            }
                        }
                        if (letterCount > 3) {
                            continue;
                        }

                        if (removeStopWords) {
                            if (StopWordsFilter.isStopWordOrNumber(subword)) {
                                continue;
                            }
                        }

                        indexContentBuilder.add(subword);
                    }
                }
            }
        } catch (Exception ex) {
            logger.error("", ex);
        }
        return indexContentBuilder;
    }

    public String cleanTxtStr(Document doc, String docid, boolean removeStopWords){
        String cleanContent = rawTxt(doc, docid);
        List<String> terms = clean2tokens(cleanContent, removeStopWords);
        return StringUtils.join(terms, " ");
    }

//    public String cleanTxtStr(String doc, String docid, boolean removeStopWords){
//        String cleanContent = rawTxt(doc, docid);
//        List<String> terms = clean2tokens(cleanContent, removeStopWords);
//        return StringUtils.join(terms, " ");
//    }

    public String[] clean2ContentTitle(Document doc, String docid, boolean removeStopWords){
        String[] titleContent = contentNTitle(doc, docid);
        if (titleContent == null){
            return null;
        }
        String title = titleContent[0];
        String content = titleContent[1];
        String titleStr, contentStr;
        List<String> titleTerms;
        if (title.length() > 1){
            titleTerms = clean2tokens(title, removeStopWords);
            titleStr = StringUtils.join(titleTerms, " ");
        } else {
            titleStr = "";
        }
        List<String> contentTerms = clean2tokens(content, removeStopWords);
        contentStr = StringUtils.join(contentTerms, " ");
        return new String[]{titleStr, contentStr};
    }

    public String cleanTxtStr(String doc, String docid, boolean removeStopWords){
        String cleanContent = rawTxt(doc);
        List<String> terms = clean2tokens(cleanContent, removeStopWords);
        return StringUtils.join(terms, " ");
    }

    public List<String> clean2tokens(String doc, String docid, boolean removeStopWords){
        String cleanContent = rawTxt(doc);
        return clean2tokens(cleanContent, removeStopWords);
    }

    public String rawTxt(Document doc, String docid) {
        String title = doc.title();
        Element body = doc.body();
        if (body == null) {
            logger.error("body of the webpage is null: " + docid);
            return null;
        }
        String core = doc.body().text();
        String cleanContent = (title != null ? title : "") + "\n" + core;
        return rawTxt(cleanContent);
    }

    private String[] contentNTitle(Document doc, String docid) {
        String title = doc.title();
        Element body = doc.body();
        if (body == null) {
            logger.error("body of the webpage is null: " + docid);
            return null;
        }
        String core = doc.body().text();
        String[] titlecontent = new String[2];
        titlecontent[1]  = rawTxt(core);
        String cleanTitle;

        if (title != null) {
            cleanTitle = rawTxt(title).replaceAll("\n", " ");
        } else {
            cleanTitle = "";
        }
        titlecontent[0] = cleanTitle;
        return titlecontent;
    }

    private String rawTxt(String doc) {
        // replace left-over tags
        doc = doc.replaceAll("<.*?[/]{0,1}>", "");
        return doc;
    }

    public List<String> clean2tokens(Document doc, String docid, boolean removeStopWords) {
        String cleanContent = rawTxt(doc, docid);
        if (cleanContent == null) {
            return null;
        }
        return clean2tokens(cleanContent, removeStopWords);
    }

    public List<String> clean2sentences(Document doc, String docid, boolean removeStopWords) {
        String cleanContent = rawTxt(doc, docid);
        if (cleanContent == null) {
            return null;
        }
        return clean2sentences(cleanContent, removeStopWords);
    }

    public List<String> clean2tokens(Document doc, boolean removeStopWords) {
        return clean2tokens(doc, "unknowncwid", removeStopWords);
    }

    public static String porterStemming(String term) {
        porterStemmer.setCurrent(term);
        porterStemmer.stem();
        return porterStemmer.getCurrent();
    }
}
