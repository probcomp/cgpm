
def check_entries_vald(valid_entries_list, user_entries_list):
	all_valid = True
	invalid_entries = []
	for user_entry in user_entries_list:
		if user_entry not in valid_entries_list:
			invalid_entries.append(user_entry)
			all_valid = False

	if not all_valid:
		errmsg = "Invalid entries: %s. Valid entries are: %s." % \
					(str(invalid_entries), str(valid_entries_list))
		return False, errmsg
	else:
		return True, []