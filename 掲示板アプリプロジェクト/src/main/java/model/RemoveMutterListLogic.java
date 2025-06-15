package model;

import dao.MuttersDAO;

public class RemoveMutterListLogic {
	public void remove(int id) {
		MuttersDAO dao = new MuttersDAO();
		dao.remove(id);
	}
}
