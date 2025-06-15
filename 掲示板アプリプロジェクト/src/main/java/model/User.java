package model;

import java.io.Serializable;

public class User implements Serializable {
	private String name;
	private String pass;
	
	public User() {
		super();
	}
	public User(String name, String pass) {
		super();
		this.name = name;
		this.pass = pass;
	}
	
	public String getName() {
		return name;
	}
	public String getPass() {
		return pass;
	}
}
