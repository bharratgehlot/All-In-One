class Book
{
    private String name;
    private int price, no_of_pages;

    Book(){
        no_of_pages = 20;
        System.out.println("Default Constructor called");

    }
    
    Book(String name,int price,int no_of_pages){
        this.name = name ;
        this.price = price ;
        this.no_of_pages = no_of_pages;
        
    }
    Book(Book M){
        this.name = M.name;
        this.price = M.price;
        this.no_of_pages = M.no_of_pages;
    }


    String getname(){
        return name;
    }
    int getprice(){
        return price;
    }
    int get_no_of_pages(){
        return no_of_pages;
    }

    
}

public class ConstructorOver {

    public static void main(String [] args){
        
        Book b = new Book();
        System.out.println("Name is "+b.getname());
        System.out.println("Price is "+b.getprice());
        System.out.println("No of Pages is "+b.get_no_of_pages());

        Book b1 = new Book("Let Us c",450,500);
        System.out.println("Name is "+b1.getname());
        System.out.println("Price is "+b1.getprice());
        System.out.println("No of Pages is "+b1.get_no_of_pages());

        // Calling coyp constructor
        Book b3 = new Book(b1);
        System.out.println("Name is "+b3.getname());
        System.out.println("Price is "+b3.getprice());
        System.out.println("No of Pages is "+b3.get_no_of_pages());
    }
    
}
